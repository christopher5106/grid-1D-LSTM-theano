'''
Recurrent network example.  Trains a 1D Grid LSTM network to learn
XOR function. The network can then be used to generate sequence using a short binary sequence as seed.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


generation_phrase = [0,1,0,0,1,1,0,1] #This phrase will be used as seed to generate text.

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_iterations', type=int, default=15000000, help='Number of iterations')
parser.add_argument('--seq_length', type=int, default=8, help='Sequence length')
parser.add_argument('--n_hidden', type=int, default=512, help='Number of units in the two hidden (LSTM) layers')
parser.add_argument('--learning_rate', type=float, default=.06, help='Optimization learning rate')
#parser.add_argument('--grad_clip', type=int, default=100, help='All gradients above this will be clipped')
parser.add_argument('--print_freq', type=int, default=10000, help='How often should we check the output?')
#parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the net')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_steps', type=int, default=70, help='Number of steps')
args = parser.parse_args()

print("Parameters:")
print(args)

args.print_batch_freq = args.print_freq / args.batch_size + 1


def gen_data(seq_length=args.seq_length, batch_size = args.batch_size):
    x = np.random.randint(2, size=(batch_size,seq_length))
    y = x.sum(axis=1) % 2
    return x, y

def main():
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None,args.seq_length))
    l_in_zero = lasagne.layers.InputLayer(shape=(None, args.num_steps, 1))

    l_lin = lasagne.layers.DenseLayer(l_in, num_units = args.n_hidden, nonlinearity = None)

    # We now build the LSTM layer which takes l_in as the hidden init
    # l_in_zero is used to provide zero input into the network (to implement GRID lstm)

    l_forward = lasagne.layers.LSTMLayer(
       l_in_zero, args.n_hidden, 
        nonlinearity=lasagne.nonlinearities.tanh, hid_init = l_lin, only_return_final=True)

    

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the linear layer 
    # to reduce the umber of dimensions to two, and then through softmax layer to 
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, 2)
    l_lin_out = lasagne.layers.DenseLayer(l_forward, num_units = 2, nonlinearity = None)
    l_out = lasagne.layers.DenseLayer(l_lin_out, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
    accuracy = lasagne.objectives.categorical_accuracy(network_output,target_values).mean()


    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, args.learning_rate)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, allow_input_downcast=True)
    compute_accuracy = theano.function([l_in.input_var, l_in_zero.input_var, target_values], 
        accuracy, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([l_in.input_var, l_in_zero.input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    def try_it_out(N=50):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        The function works in three steps:
        1. We use the LSTM to predict the next character and store it in a (dynamic) list sample_ix. This is done by using the 'probs'
           function which was compiled above. Simply put, given the output, we compute the probabilities of the target and pick the one 
           with the highest predicted probability. 
        2. Once this character has been predicted, we construct a new sequence using all but first characters of the 
           provided string and the predicted character. This sequence is then used to generate yet another character.
           This process continues for "N" characters. 
        To make this clear, let us again look at a concrete example. 
        Assume that SEQ_LENGTH = 5 and generation_phrase = [1,0,0,1,1]. 
        The next character is then predicted (as explained in step 1). 
        Assume that this character was '0'. We then construct a new sequence using the last 4 (=SEQ_LENGTH-1) characters of the previous
        sequence [0,0,1,1] , and the predicted symbol '0'. This new sequence is then used to compute the next character and 
        the process continues.
        '''

        assert(len(generation_phrase)==args.seq_length)
        sample_ix = []
        x = np.zeros((1,len(generation_phrase)),dtype='int32')
        x [0,:] = generation_phrase
        x_zero = np.zeros((len(x),args.num_steps,1),dtype='int32')  # size (batch_size,num_steps,1)

        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x,x_zero).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[0,:args.seq_length - 1] = x[0,1:]
            x[0,args.seq_length - 1] = ix

        random_snippet = generation_phrase + sample_ix
        print("----\n %s \n----" % random_snippet)


    
    print("Training ...")
    #print("Seed used for text generation is: " + ''.join(str(char) for char in generation_phrase))
    print("The average loss and accuracy will be printed every {} iterations".format(args.print_batch_freq*args.batch_size))
    num_batch_print_iter = args.num_iterations / args.batch_size / args.print_batch_freq + 1
    act_num_batches =  num_batch_print_iter * args.print_batch_freq 
    all_cost = np.zeros((act_num_batches))
    all_accuracy = np.zeros((act_num_batches))
    try:
        for it_out in xrange(num_batch_print_iter):
            #try_it_out() 
            
            for it_in in range(args.print_batch_freq):
                x,y = gen_data()
                x_zero = np.zeros((args.batch_size,args.num_steps,1),dtype='int32')

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
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
