from __future__ import print_function

import os
import sys
import timeit
import gzip, cPickle, glob

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
#from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

import argparse
from pyroc import *

def sigmoid(x):
    return T.nnet.sigmoid( (x - 5.) )

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def Load_data(dataset):
    print ('... loading data')
    Indim, AE_set, train_set, valid_set, test_set = cPickle.load( gzip.open(dataset, 'rb') )
    AE_set_x = theano.shared(numpy.asarray(AE_set, dtype=theano.config.floatX), borrow=True)
    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    rval = [AE_set_x, (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return Indim, rval

def tee_stdout ( f_out, msg ):
    f_out.write(msg)
    f_out.write("\n")
    fout.flush()
    print(msg)

class dA(object):
    def __init__(self, rng, input=None, n_visible=None, n_hidden=None, W=None, bhid=None, bvis=None ):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.x         = input
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        W_bound= numpy.sqrt(6. / (n_visible + n_hidden))  #
        self.W = theano.shared( numpy.asarray(W_bound *
                                rng.standard_normal(size=(n_visible, n_hidden)),
                                dtype=theano.config.floatX), borrow=True )
        bvis_values = numpy.zeros( n_visible, dtype=theano.config.floatX )
        bvis = theano.shared( value=bvis_values,borrow=True)
        bhis_values = numpy.zeros(n_hidden, dtype=theano.config.floatX )
        bhid = theano.shared( value=bhis_values, name='b', borrow=True )
        self.b       = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.params  = [self.W, self.b, self.b_prime]
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) * input
    def get_hidden_values(self, input):
        return T.nnet.relu(T.dot(input, self.W) + self.b)
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    def get_cost_updates(self, corruption_level, pre_learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        cost    = T.mean( -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1) )
        gparams = T.grad(cost, self.params)
        updates = [(param, param - pre_learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return (cost, updates)
    def pretraining_function(self, AE_set_x):
        index = T.lvector('index')
        corruption_level = T.scalar('corruption')
        pre_learning_rate= T.scalar('lr')
        cost, updates = self.get_cost_updates(corruption_level, pre_learning_rate)
        fn = theano.function(inputs=[index, corruption_level,   pre_learning_rate],
                             outputs=cost,
                             updates=updates,
                             givens = {self.x: AE_set_x[index]})
        return fn

####
class LeNetConvPoolLayer1(object):
    def __init__(self, rng, input, init_W, init_b, filter_shape, image_shape):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        if init_W ==None:
            fan_in  = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))  #
            init_W = theano.shared( numpy.asarray(W_bound * 
                                    rng.standard_normal(size=filter_shape), 
                                    dtype=theano.config.floatX), borrow=True )
        if init_b ==None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX) 
            init_b   = theano.shared(value=b_values, borrow=True)
        self.W = init_W
        self.b = init_b
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        self.output = T.nnet.relu( conv_out + self.b.dimshuffle('x', 0, 'x', 'x') )
        self.params = [self.W, self.b]

class LeNetConvPoolLayer2(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in  = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))
        W_bound = 4.* numpy.sqrt(6. / (fan_in + fan_out)) 
        self.W = theano.shared( W_bound * numpy.ones(filter_shape, dtype=theano.config.floatX) ,borrow=True) 
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        pooled_out = pool.pool_2d(
#        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False
        )
        b_values = -5 * numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b   = theano.shared(value=b_values, borrow=True)
        self.output = pooled_out + self.b.dimshuffle('x', 0, 'x','x')
        self.params = [self.W, self.b]
        self.p_y_given_x = T.nnet.sigmoid( self.output.flatten() )
    def negative_log_likelihood(self, y):
        return -T.mean( T.cast(y, 'float32')* T.log(self.p_y_given_x) + (1-T.cast(y, 'float32'))*T.log(1-self.p_y_given_x) )

class CNN(object):
    def __init__(self, rng, nkerns, batch_size, in_dim, filtsize, init_W, init_b):
        self.layers = []
        self.params = []
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        layer1_input = self.x.reshape((batch_size, 1, in_dim[0], in_dim[1]))
        layer1 = LeNetConvPoolLayer1(
            rng,
            input=layer1_input,
            init_W=init_W,
            init_b=init_b,
            image_shape =(batch_size, 1, in_dim[0], in_dim[1]),
            filter_shape=(nkerns[0],  1, filtsize[0], filtsize[1]))
        self.layers.append(layer1)
        layer2_input = layer1.output.reshape((batch_size, 1,nkerns[0], in_dim[1]))
        layer2 = LeNetConvPoolLayer2(
            rng,
            input=layer2_input,
            image_shape =(batch_size, 1,nkerns[0], in_dim[1]),
            filter_shape=(1,  1,nkerns[0], 1),
            poolsize    =[1, in_dim[1]])
        self.layers.append(layer2)
        self.L1 = abs(layer1.W.flatten()).sum() + abs(layer2.W.flatten()).sum()
        self.L2 = (layer1.W.flatten()**2).sum() + (layer2.W.flatten()**2).sum()
        self.params = [ param for layer in self.layers for param in layer.params ]
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        self.finetune_cost = layer2.negative_log_likelihood(self.y)
    def build_finetune_functions(self, datasets, batch_size, learning_rate, L1_param, L2_param, mom):
        (train_set_x, train_set_y) = datasets[1]
        (valid_set_x, valid_set_y) = datasets[2]
        (test_set_x, test_set_y)   = datasets[3]
        index = T.lvector('index')
        cost = self.finetune_cost + L1_param *self.L1 + L2_param * self.L2
        gparams = T.grad( cost, self.params)
        updates1 = OrderedDict()
        for param, gparam, gparam_mom in zip(self.params, gparams, self.gparams_mom):
            updates1[gparam_mom] = mom * gparam_mom - learning_rate * gparam
            updates1[param] = param + updates1[gparam_mom]
        train_fn = theano.function(
            inputs =[index],
            outputs=self.finetune_cost,
            updates=updates1,
            givens={ self.x: train_set_x[index],
                     self.y: train_set_y[index]}  )
        # error check
        valid_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: valid_set_x[index]} )
        valid_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: valid_set_y[index] } )
        # performance check : error rate, sensitivity, specificity, auc
        test_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: test_set_x[index]} )
        test_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: test_set_y[index] } )
        def getVals( fn, IDX, n_exp, batch_size ):
            vals = list()
            n_batches = n_exp/ batch_size
            cnt  = int(batch_size/float(n_exp))
            resid= n_exp - (n_batches *batch_size)
            for i in range(n_batches):
                vals+= fn(IDX[i*batch_size:(i+1)*batch_size]).tolist()
            if cnt <  1 and resid !=0:
                val  = fn(IDX[((n_batches-1)*batch_size+resid):((n_batches*batch_size)+resid)]).tolist()
                vals+= val[(batch_size-resid):batch_size]
	    if cnt >= 1 and resid !=0:
                IDX_ = IDX
                for i in range(cnt):
                    IDX_ = numpy.concatenate((IDX_, IDX))
                val = fn(IDX_[0: batch_size])
                vals += numpy.array(val)[range(n_exp)].tolist()
            return vals
        n_valid_exp = valid_set_x.get_value(borrow=True).shape[0]
        n_test_exp  =  test_set_x.get_value(borrow=True).shape[0]
        def valid_check():
            idx = numpy.random.permutation(range(n_valid_exp))
            valid_y    = getVals( valid_y_fn,   idx, n_valid_exp, batch_size )
            valid_pred = getVals( valid_pred_fn,idx, n_valid_exp, batch_size )
            return valid_y, valid_pred
        def test_check():
            idx = numpy.random.permutation(range(n_test_exp))
            test_y     = getVals( test_y_fn,    idx, n_test_exp, batch_size )
            test_pred  = getVals( test_pred_fn, idx, n_test_exp, batch_size )
            return test_y, test_pred
        return train_fn, valid_check, test_check

def evaluate_lenet5(datasets=None, AE=None, learning_rate=None, pre_learning_rate=None,
                    nkerns=None,   batch_size=None, 
                    L1_param=None, L2_param=None, mom=None,
                    in_dim=None,    filtsize=None, corruption_level=None):

    AE_set_x		     = datasets[0]
    train_set_x, train_set_y = datasets[1]
    valid_set_x, valid_set_y = datasets[2]
    test_set_x, test_set_y   = datasets[3]

    tee_stdout ( fout, ('... Start AutoEncoder training' ))
    numpy_rng = numpy.random.RandomState(2**30)

    n_train_exp     = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train_exp / batch_size


    if AE ==False:
        init_W = None
        init_b = None
    else:
        n_AEtrain_exp, n_ins = AE_set_x.get_value(borrow=True).shape
        n_AEtrain_batches    = n_AEtrain_exp / batch_size

        AE_input = T.matrix('AE_input')
        sda = dA(rng=numpy_rng, input=AE_input, n_visible=n_ins, n_hidden=nkerns[0]) ###
        pretraining_fn  = sda.pretraining_function(AE_set_x=AE_set_x)
    
        for epoch in range(10):
            c= list()
            IDX = numpy.random.permutation(range(n_AEtrain_exp))
            for i in range(n_AEtrain_batches):
                c.append( pretraining_fn(IDX[i*batch_size:(i+1)*batch_size], corruption_level, pre_learning_rate  ))
            tee_stdout ( fout, ('... Pre-training layer, epoch %d, cost %.3f'%(epoch, numpy.mean(c)) ))

        w1 = sda.W.get_value().T
        new_w1 = list()
        for i in range(w1.shape[0]):
            new_w1.append( w1[i].flatten()[::-1] )
        new_w1 = numpy.array(new_w1)

        init_W = theano.shared(numpy.asarray(new_w1.reshape((nkerns[0],  1, filtsize[0], filtsize[1])) ),borrow=True)
        init_b = theano.shared(numpy.asarray(sda.b.get_value()),borrow=True)


    tee_stdout ( fout, ('... building the model' ))
    cnn = CNN(rng=numpy_rng, nkerns=nkerns, batch_size=batch_size,
          in_dim=in_dim, filtsize=filtsize, init_W=init_W, init_b=init_b)
    ###############
    # TRAIN MODEL #
    ###############
    tee_stdout ( fout, ('... training' ))
    train_model, valid_model, test_model  = cnn.build_finetune_functions(datasets=datasets,
                                                                         batch_size=batch_size, 
                                                                         learning_rate=learning_rate, 
                                                                         L1_param=L1_param, 
                                                                         L2_param=L2_param, 
                                                                         mom=mom)

    patience = 200 * n_train_batches
    patience_increase = 2. 
    improvement_threshold = 1.002
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_score = -1.
    best_iter  = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    n_epochs=10000000
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        permutation = numpy.random.permutation(n_train_exp)
        for minibatch_index in range(n_train_batches):

            batch_begin = minibatch_index * batch_size
            batch_end = batch_begin + batch_size

            minibatch_avg_cost = train_model(permutation[batch_begin:batch_end])

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                #### VALIDATION CHECK
                valid_y, valid_pred = valid_model()
                if sum( map(math.isnan, valid_pred) ) ==0 :
                    this_auc                    = ROCData( zip(valid_y, valid_pred )).auc()
                    valid_y, valid_pred         = numpy.array(valid_y), numpy.array(valid_pred)
                    this_validation_sensitivity = numpy.mean( valid_pred[numpy.equal(valid_y, 1)] >.5 )
                    this_validation_specificity = numpy.mean( valid_pred[numpy.equal(valid_y, 0)]<=.5 )
                    this_validation_precision   = numpy.mean( numpy.equal(valid_y[valid_pred > .5],1) )
                    this_validation_F1          = 2./(1./this_validation_precision + 1./this_validation_sensitivity)
                    if math.isnan(this_validation_F1) ==False :
                        this_validation_score   = this_validation_F1
                    else:
                        this_validation_F1, this_validation_score = 0.,0.

                else:
                    this_auc, this_validation_score, this_validation_sensitivity, this_validation_specificity, this_validation_F1 = 0.,0.,0.,0.,0.

                msg = 'epoch %i minibatch %i/%i sensitivity %.2f specificity %.2f AUC %f F1 %f' % (
                        epoch, 
                        minibatch_index + 1, 
                        n_train_batches,
                        this_validation_sensitivity *100.,
                        this_validation_specificity *100.,
                        this_auc *100.,
                        this_validation_F1 *100. )

                tee_stdout(fout, msg)
                if this_validation_score > best_validation_score:
                    if this_validation_score > best_validation_score * improvement_threshold:
                        patience = min(max(patience, iter * patience_increase), 10000)
                    best_validation_score = this_validation_score
                    best_iter = iter

                    #### TEST CHECK
                    test_y, test_pred = test_model()
                    if sum( map(math.isnan, test_pred) ) ==0 :
                        test_auc          = ROCData( zip(test_y, test_pred)).auc()
                        test_y, test_pred = numpy.array(test_y), numpy.array(test_pred)
                        test_sensitivity  = numpy.mean(test_pred[numpy.equal(test_y, 1)] >.5 )
                        test_specificity  = numpy.mean(test_pred[numpy.equal(test_y, 0)] <=.5)
                        test_precision    = numpy.mean( numpy.equal(test_y[test_pred > .5],1) )
                        test_F1           = 2./(1./test_precision + 1./test_sensitivity)
                        if math.isnan(test_F1) ==True :
                            test_F1    = 0.
                    else:
                        test_auc, test_sensitivity, test_specificity, test_F1 = 0.,0.,0.,0.,0.

                    msg = '     test epoch %i sensitivity %.2f specificity %.2f AUC %.2f F1 %.2f\n' % (
                         epoch, 
                         test_sensitivity *100., 
                         test_specificity *100., 
                         test_auc *100.,
                         test_F1  *100. )

                    tee_stdout(fout, msg)

                    # save model
                    #cPickle.dump(cnn, gzip.open(saveFile+'.pkl.gz', 'wb'),protocol=2)
                    cPickle.dump([numpy.array(param.get_value(borrow=True)) for param in cnn.params], gzip.open(saveFile+'.params.pkl.gz','wb'),protocol=2)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    msg= 'Optimization complete.. Best validation score of %f obtained at iteration %i, with test sensitivity %.2f specificity %.2f AUC %.2f F1 %.2f\nThe code for file ran for %.2fm' % (
          best_validation_score * 100., best_iter + 1, test_sensitivity *100., test_specificity *100., test_auc *100., test_F1 *100., (end_time - start_time)/ 60.)

    tee_stdout(fout, msg)


if __name__ == '__main__':
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--Input',       default='../02_PreProc/Data/AUTsummary_hg19_AE_30SNP.pkl.gz', help='Input pkl.gz file')
    parser.add_argument('--learning_rate',type=float, default=0.5, help='learning rate')
    parser.add_argument('--pre_learning_rate',type=float, default=0.5, help='pre_learning rate')
    parser.add_argument('--n_kerns',     type=int,    default=20,  help='number of kernel')
    parser.add_argument('--batch_size',  type=int,    default=100,  help='batch size')
    parser.add_argument('--L1_param',    type=float,  default=5.,  help='L1 regularization parameter')
    parser.add_argument('--L2_param',    type=float,  default=0.,  help='L2 regularization parameter')
    parser.add_argument('--mom',         type=float,  default=0.,  help='momentum')
    parser.add_argument('--corruption_level',type=float,default=0.,help='corruption_level')
    args=parser.parse_args()

    Name = args.Input.split('/')[-1].split('_')[0]
    out_dir = './'+Name+'_out'

    if os.path.exists(out_dir)==False:
        os.system('mkdir '+out_dir)

    Indim, datasets = Load_data(args.Input)
    n_feature, n_snp = Indim

    for AE_ in [True,False]:
        TMP = map(str, [ Name,
                        'AE', AE_,
                        'learningRate', args.learning_rate,
                        'preLearningRate',args.pre_learning_rate,
                        'nKerns',      args.n_kerns,
                        'batchSize',   args.batch_size,
                        'L1Param',     args.L1_param,
                        'L2Param',     args.L2_param,
                        'mom',         args.mom,
                        'corruptionLevel',args.corruption_level])

        saveFile=out_dir+'/'+'_'.join(TMP)
        fout=open(saveFile+'.log','w')
        tee_stdout( fout, ('Write file... '+saveFile))
        tee_stdout( fout, ('Start... '     +args.Input))

        evaluate_lenet5(datasets	 = datasets, 
                        AE           	 = AE_, 
                        learning_rate	 = args.learning_rate,
                        pre_learning_rate=args.pre_learning_rate, 
                        nkerns	 	 = [args.n_kerns],   
                        batch_size	 = args.batch_size, 
                        L1_param	 = args.L1_param, 
                        L2_param	 = args.L2_param ,
                        mom		 = args.mom,
                        in_dim	 	 = [n_feature, n_snp],
                        filtsize	 = [n_feature,1],
                        corruption_level=args.corruption_level)
