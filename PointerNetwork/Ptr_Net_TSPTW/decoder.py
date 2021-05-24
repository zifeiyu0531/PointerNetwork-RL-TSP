import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

distr = tf.contrib.distributions


# RNN decoder for pointer network
class Pointer_decoder(object):

    def __init__(self, encoder_output, config):
        #######################################
        ########## Reference vectors ##########
        #######################################

        self.encoder_output = encoder_output  # Tensor [Batch size x time steps x cell.state_size] to attend to
        self.h = tf.transpose(encoder_output, [1, 0, 2])  # [Batch size x time steps x cell.state_size] to [time steps x Batch size x cell.state_size]

        ############################
        ########## Config ##########
        ############################

        batch_size = encoder_output.get_shape().as_list()[0]  # batch size
        self.seq_length = encoder_output.get_shape().as_list()[1]  # sequence length
        n_hidden = encoder_output.get_shape().as_list()[2]  # num_neurons

        self.inference_mode = config.inference_mode  # True for inference, False for training
        self.temperature = config.temperature  # temperature parameter
        self.C = config.C  # logit clip

        ##########################################
        ########## Decoder's parameters ##########
        ##########################################

        # Variables initializer
        initializer = tf.contrib.layers.xavier_initializer()

        # Decoder LSTM cell        
        self.cell = LSTMCell(n_hidden, initializer=initializer)

        # Attending mechanism
        with tf.variable_scope("glimpse") as glimpse:
            self.W_ref_g = tf.get_variable("W_ref_g", [1, n_hidden, n_hidden], initializer=initializer)
            self.W_q_g = tf.get_variable("W_q_g", [n_hidden, n_hidden], initializer=initializer)
            self.v_g = tf.get_variable("v_g", [n_hidden], initializer=initializer)

        # Pointing mechanism
        with tf.variable_scope("pointer") as pointer:
            self.W_ref = tf.get_variable("W_ref", [1, n_hidden, n_hidden], initializer=initializer)
            self.W_q = tf.get_variable("W_q", [n_hidden, n_hidden], initializer=initializer)
            self.v = tf.get_variable("v", [n_hidden], initializer=initializer)

        ######################################
        ########## Decoder's output ##########
        ######################################

        self.log_softmax = []  # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = []  # store task list for reward
        self.attending = []  # for vizualition
        self.pointing = []  # for vizualition

        ########################################
        ########## Initialize process ##########
        ########################################

        # Start from depot
        self.depot_position = tf.constant(self.seq_length - 1, shape=[batch_size])
        self.positions.append(self.depot_position)

        # Keep track of current city
        self.current_task = tf.one_hot(self.depot_position, self.seq_length)

        # Keep track of visited cities  [0,0,0,...,1]
        self.mask = tf.one_hot(self.depot_position, self.seq_length)

    def loop_decode(self, decoder_initial_state):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Decoder initial input is depot (start)
        decoder_first_input = tf.gather(self.h, self.depot_position)[0]

        # Loop the decoding process and collect results
        s, i = decoder_initial_state, decoder_first_input
        for step in range(self.seq_length - 1):
            s, i = self.decode(s, i, step)

        # Stack visited indices
        self.positions = tf.stack(self.positions, axis=1)  # [Batch,seq_length]

        # Sum log_softmax over output steps
        self.log_softmax = tf.add_n(self.log_softmax)  # [Batch,seq_length-1]

        # Stack attending & pointing distribution
        self.attending = tf.stack(self.attending, axis=1)  # [Batch,seq_length-1,seq_length]
        self.pointing = tf.stack(self.pointing, axis=1)  # [Batch,seq_length-1,seq_length]

        # Return stacked lists of visited_indices and log_softmax for backprop
        return self.positions, self.log_softmax, self.attending, self.pointing

    # One pass of the decode mechanism
    def decode(self, prev_state, prev_input, timestep):
        with tf.variable_scope("loop"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()

            # Run the cell on a combination of the previous input and state
            output, state = self.cell(prev_input, prev_state)

            # Attention mechanism (256, 20)
            masked_scores = self.attention(self.encoder_output, output)
            # Multinomial distribution
            prob = distr.Categorical(masked_scores)

            # Sample from distribution
            position = prob.sample()
            position = tf.cast(position, tf.int32)
            self.positions.append(position)

            # Store log_prob for backprop
            self.log_softmax.append(prob.log_prob(position))

            # Update current task and mask
            self.current_task = tf.one_hot(position, self.seq_length)
            self.mask = self.mask + self.current_task

            # Retrieve decoder's new input
            new_decoder_input = tf.gather(self.h, position)[0]

            return state, new_decoder_input

    # From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    # predict a distribution over next decoder input
    def attention(self, ref, query):

        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, self.W_ref_g, 1, "VALID",
                                     name="encoded_ref_g")  # [Batch size, seq_length, n_hidden]
        encoded_query_g = tf.expand_dims(tf.matmul(query, self.W_q_g, name="encoded_query_g"),
                                         1)  # [Batch size, 1, n_hidden]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1],
                                 name="scores_g")  # [Batch size, seq_length]

        # Attend to current task and tasks to apply only (Apply mask)
        attention_g = tf.nn.softmax(scores_g - 100000000. * (self.mask - self.current_task), name="attention_g")
        self.attending.append(attention_g)

        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g, 2))
        glimpse = tf.reduce_sum(glimpse, 1) + query

        # Pointing mechanism with 1 glimpse
        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID",
                                   name="encoded_ref")  # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(glimpse, self.W_q, name="encoded_query"),
                                       1)  # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1],
                               name="scores")  # [Batch size, seq_length]
        if self.inference_mode:
            scores = scores / self.temperature  # control diversity of sampling (inference mode)
        scores = self.C * tf.tanh(scores)  # control entropy

        # Point to cities to visit only (Apply mask)
        masked_scores = scores - 100000000. * self.mask  # [Batch size, seq_length]
        pointing = tf.nn.softmax(masked_scores, name="attention")  # [Batch size, Seq_length]
        self.pointing.append(pointing)

        return masked_scores