'''
Recurrent network example.  Trains a 1D Grid LSTM network to learn
XOR answers.
'''

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=15000000, help='Number of iterations')
parser.add_argument('--bits', type=int, default=8, help='Number of bits in the input strings')
parser.add_argument('--hidden', type=int, default=512, help='Number of units in the two hidden (LSTM) layers')
parser.add_argument('--learning_rate', type=float, default=.06, help='Optimization learning rate')
#parser.add_argument('--grad_clip', type=int, default=100, help='All gradients above this will be clipped')
parser.add_argument('--print_freq', type=int, default=10000, help='How often should we check the output?')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--layers', type=int, default=70, help='Number of layers')
args = parser.parse_args()

print("Parameters:")
print(args)
args.print_batch_freq = args.print_freq / args.batch_size + 1

def gen_data(bits=args.bits, batch_size = args.batch_size):
    x = np.random.randint(2, size=(batch_size,bits))
    y = x.sum(axis=1) % 2
    return x, y

print("Building network ...")

l_in = lasagne.layers.InputLayer(shape=(None,args.bits))
l_in_zero = lasagne.layers.InputLayer(shape=(None, args.layers, 1))
l_lin = lasagne.layers.DenseLayer(l_in, num_units = args.hidden, nonlinearity = None)

l_forward = lasagne.layers.LSTMLayer(
   l_in_zero, args.hidden,
    nonlinearity=lasagne.nonlinearities.tanh, hid_init = l_lin, only_return_final=True)

l_lin_out = lasagne.layers.DenseLayer(l_forward, num_units = 2, nonlinearity = None)
l_out = lasagne.layers.DenseLayer(l_lin_out, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

target_values = T.ivector('target_output')

network_output = lasagne.layers.get_output(l_out)

cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
accuracy = lasagne.objectives.categorical_accuracy(network_output,target_values).mean()

all_params = lasagne.layers.get_all_params(l_out,trainable=True)

print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, args.learning_rate)

print("Compiling functions ...")
train = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, allow_input_downcast=True)
compute_accuracy = theano.function([l_in.input_var, l_in_zero.input_var, target_values],
    accuracy, allow_input_downcast=True)

probs = theano.function([l_in.input_var, l_in_zero.input_var],network_output,allow_input_downcast=True)

print("Training ...")
print("The average loss and accuracy will be printed every {} iterations".format(args.print_batch_freq*args.batch_size))
num_batch_print_iter = args.iterations / args.batch_size / args.print_batch_freq + 1
act_num_batches =  num_batch_print_iter * args.print_batch_freq
all_cost = np.zeros((act_num_batches))
all_accuracy = np.zeros((act_num_batches))

for it_out in xrange(num_batch_print_iter):
    for it_in in range(args.print_batch_freq):
        x,y = gen_data()
        x_zero = np.zeros((args.batch_size,args.layers,1),dtype='int32')

        batch_iter = it_out * args.print_batch_freq + it_in + 1
        batch_cost = train(x, x_zero, y)
        batch_accuracy = compute_accuracy(x,x_zero,y)

        all_cost[batch_iter - 1] = batch_cost
        all_accuracy[batch_iter - 1] = batch_accuracy
    start_index = it_out * args.print_batch_freq
    end_index = (it_out + 1) * args.print_batch_freq
    av_cost = all_cost[start_index:end_index].mean()
    av_accuracy = all_accuracy[start_index:end_index].mean()
    np.savetxt('cost.txt', all_cost[:end_index], delimiter=',')  #average in batch
    np.savetxt('accuracy.txt', all_accuracy[:end_index], delimiter=',')

    print("Iteration {} average loss = {} average accuracy = {}".format(batch_iter*args.batch_size,
        av_cost,av_accuracy))
