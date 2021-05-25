import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Critic(object):

    def __init__(self, config):
        self.config = config

        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.hidden_dim  # dimension of hidden states (LSTM cell)
        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer

        # Baseline setup
        self.init_baseline = self.max_length / 2.

        # Training config
        self.is_training = config.training_mode

    def predict_rewards(self, input_):
        with tf.variable_scope("encoder"):
            with tf.variable_scope("embedding"):
                # Embed input sequence
                W_embed = tf.get_variable("weights", [1, self.input_dimension, self.input_embed],
                                          initializer=self.initializer)
                embedded_input = tf.nn.conv1d(input_, W_embed, 1, "VALID", name="embedded_input")
                # Batch Normalization
                embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training,
                                                               name='layer_norm', reuse=None)

            with tf.variable_scope("dynamic_rnn"):
                # Encode input sequence
                cell1 = LSTMCell(self.num_neurons, initializer=self.initializer)
                # Return the output activations and last hidden state (c,h) as tensors.
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embedded_input, dtype=tf.float32)
                # frame = tf.reduce_mean(encoder_output, 1)
                frame = encoder_state[0]  # [Batch size, Num_neurons]

            # Glimpse
            with tf.variable_scope("glimpse"):
                self.W_ref_g = tf.get_variable("W_ref_g", [1, self.num_neurons, self.num_neurons],
                                               initializer=self.initializer)
                self.W_q_g = tf.get_variable("W_q_g", [self.num_neurons, self.num_neurons],
                                             initializer=self.initializer)
                self.v_g = tf.get_variable("v_g", [self.num_neurons], initializer=self.initializer)

                # Attending mechanism
                encoded_ref_g = tf.nn.conv1d(encoder_output, self.W_ref_g, 1, "VALID",
                                             name="encoded_ref_g")  # [Batch size, seq_length, n_hidden]
                encoded_query_g = tf.expand_dims(tf.matmul(frame, self.W_q_g, name="encoded_query_g"),
                                                 1)  # [Batch size, 1, n_hidden]
                scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1],
                                         name="scores_g")  # [Batch size, seq_length]
                attention_g = tf.nn.softmax(scores_g, name="attention_g")

                # 1 glimpse = Linear combination of reference vectors (defines new query vector)
                glimpse = tf.multiply(encoder_output, tf.expand_dims(attention_g, 2))
                glimpse = tf.reduce_sum(glimpse, 1)

            with tf.variable_scope("ffn"):
                # ffn 1
                h0 = tf.layers.dense(glimpse, self.num_neurons, activation=tf.nn.relu,
                                     kernel_initializer=self.initializer)
                # ffn 2
                w1 = tf.get_variable("w1", [self.num_neurons, 1], initializer=self.initializer)
                b1 = tf.Variable(self.init_baseline, name="b1")
                self.predictions = tf.squeeze(tf.matmul(h0, w1) + b1)
