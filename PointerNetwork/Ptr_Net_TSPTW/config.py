# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor input embedding')
net_arg.add_argument('--hidden_dim', type=int, default=128, help='actor LSTM num_neurons')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=16
                      , help='batch size')
data_arg.add_argument('--input_dimension', type=int, default=7, help='data dimension')
data_arg.add_argument('--max_length', type=int, default=80, help='number of task')  # this excludes depot
data_arg.add_argument('--server_load', type=int, default=5, help='server load')  # this excludes depot
data_arg.add_argument('--dir_', type=str, default='n20w100', help='Dumas benchmarch instances')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--gen_num', type=int, default=300, help='ga gen num')
train_arg.add_argument('--nb_epoch', type=int, default=4000, help='nb epoch')
train_arg.add_argument('--lr1_start', type=float, default=0.0001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=500, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')

train_arg.add_argument('--alpha', type=float, default=0.3, help='weight for load impact')
train_arg.add_argument('--beta', type=float, default=0.3, help='weight for priority impact')
train_arg.add_argument('--gama', type=float, default=0.3, help='weight for timeout impact')

train_arg.add_argument('--alpha_c', type=float, default=0.25, help='weight for cpu')
train_arg.add_argument('--alpha_o', type=float, default=0.25, help='weight for io')
train_arg.add_argument('--alpha_b', type=float, default=0.25, help='weight for bandwidth')
train_arg.add_argument('--alpha_m', type=float, default=0.25, help='weight for memory')

train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')

# Misc
misc_arg = add_argument_group('User options')

misc_arg.add_argument('--pretrain', type=str2bool, default=False, help='faster datagen for infinite speed')
misc_arg.add_argument('--inference_mode', type=str2bool, default=False,
                      help='switch to inference mode when model is trained')
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='speed1000/n20w100',
                      help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='speed1000/n20w100',
                      help='loader sub directory')
misc_arg.add_argument('--log_dir', type=str, default='summary/test', help='summary writer log directory')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_config():
    config, _ = get_config()
    print('\n')
    print('Data Config:')
    print('* Batch size:', config.batch_size)
    print('* Sequence length:', config.max_length)
    print('* Task coordinates:', config.input_dimension)
    print('\n')
    print('Network Config:')
    print('* Restored model:', config.restore_model)
    print('* Actor input embedding:', config.input_embed)
    print('* Actor hidden_dim (num neurons):', config.hidden_dim)
    print('* Actor tan clipping:', config.C)
    print('\n')
    if not config.inference_mode:
        print('Training Config:')
        print('* Nb epoch:', config.nb_epoch)
        print('* Temperature:', config.temperature)
        print('* Actor learning rate (init,decay_step,decay_rate):', config.lr1_start, config.lr1_decay_step,
              config.lr1_decay_rate)
    else:
        print('Testing Config:')
    print('* Summary writer log dir:', config.log_dir)
    print('\n')
