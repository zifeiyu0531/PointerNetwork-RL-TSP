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
data_arg.add_argument('--batch_size', type=int, default=64, help='batch size')
data_arg.add_argument('--max_length', type=int, default=22, help='number of city')
data_arg.add_argument('--input_dimension', type=int, default=2, help='coordinate of city')


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--iteration', type=int, default=50000, help='number of iteration')
train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=500, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')

train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')

# Misc
misc_arg = add_argument_group('User options')

misc_arg.add_argument('--training_mode', type=str2bool, default=True,
                      help='switch to training mode when model is training')
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='./save/actor.ckpt',
                      help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='./save/actor.ckpt',
                      help='loader sub directory')


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
    if config.training_mode:
        print('Training Config:')
        print('* iteration:', config.iteration)
        print('* Temperature:', config.temperature)
        print('* Actor learning rate (init,decay_step,decay_rate):', config.lr1_start, config.lr1_decay_step,
              config.lr1_decay_rate)
    else:
        print('Testing Config:')
    print('\n')
