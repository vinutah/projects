#!/usr/bin/python
__author__ = 'vinu joseph'

import logging
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='debug.log',level=logging.DEBUG)

import argparse

#for shuffle'ing training data
import random

#for exp function
import math

#to transform sum to do element wise for vectors?
from numpy import sum

#for executing system commands
import os

#for plotting the NEGATIVE log liklihood of the data set
import matplotlib.pyplot as plt
import numpy as np

def kFoldCrossValidation(X,K):
    for k in xrange(K):
        training   = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def readweights(weightsFile,w):
    with open(weightsFile,'r') as f:
        W = f.readlines()
        w.append(W[-1].strip())
        w.append(W[-2].strip())
        w.append(W[-3].strip())
    f.close()
    return w

def getErrors(o_list, ml_list):
    sq_error = 0
    sum_sq_error = 0
    mse = 0
    error = list()
    for i in range(len(o_list)):
        error.append(float(o_list[i]) - float(ml_list[i]))
    maxE= max(error)
    
    for i in range(len(o_list)):
        sq_error = error[i] * error[i]
        sum_sq_error += sq_error
    mSE = float(sum_sq_error) / len(o_list)
    
    return maxE, mSE

def linregr_test(X,T,s,rou):
    """
    read weights
    predict yi
    calculate error
    append to a _list

    calculate mse
    find maxError and maxMse
    """
    w = list()
    readweights(weightsFile,w)

    w1 =  float(str(w[0]))
    w2 =  float(str(w[1]))
    w3 =  float(str(w[2]))

    error   = list()   
    o_list  = list()
    ml_list = list()
 
    for xi in X:
        x_i = xi.strip().split(' ')
        
        yi = float(x_i[0])
        o_list.append(yi)
        del x_i[0]

        xi = list()
        for features in range(len(x_i)):
            xi.append(x_i[features].split(':')[-1])
        f1 =  float(str(xi[0]))
        f2 =  float(str(xi[1]))
        f3 =  float(str(xi[2]))
        yi_p = w1*f1 + w2*f2 + w3*f3
        ml_list.append(yi_p)   
     
    maxError, meanSqError = getErrors(o_list,ml_list)
 
    return maxError, meanSqError

def linregr_train(X,T,s,rou):
    logging.debug( "number of epochs       :%d", T )
    logging.debug( "variance of w or sigma :%f", s ) 
    logging.debug( "learning rate          :%f", rou )
    """find out how many features there are 
       this will determing the length of tht weight vector"""

    x_1 = X[0].strip().split(' ')

    """ zero initialize the weight vector
        len x_1 - 1 for not counting the label from the example """
    w   = [0 for i in range(len(x_1) - 1)]

    """ 
        initialize the learning rate to 1 to begin with
        the rule for SGD to have any gurantees is that 
        the learning rate is square summable sum (t=1 to infinity) r_t = infinity
        the learning rate is not summable    sum (t=1 to infinity) r_t < infinity
        r(t, rou) = rou / (1 + rou*t/c)
        here the rou and c are hyperparameters for this algorithm
    """
    t = 0

    for epoch in range(0,T+1):
        random.shuffle(X)
        for xi in X:
            """ calculate the learning rate """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            r = (rou) / (1 + (rou*(t/c)))            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
            """ read the line into this list and split on spaces """
            x_i = xi.strip().split(' ')
            
            """ get the actual value of this exmaple and delete it from the x_i list :) """
            yi = float(x_i[0])
            del x_i[0]
            
            """ load the features into a list data structure """
            xi = list()
            for features in range(len(x_i)):
                xi.append(x_i[features].split(':')[-1])
            #logging.debug(xi)
            
            """ find the w dot x for this example """
            #logging.debug( "length of xi is : %d", len(xi) ) 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            wTxi = sum( [ float(w[f]) * float(xi[f]) for f in range(len(xi)) ]  )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #logging.debug(wTxi)

            """ 
            now here's the real deal !
            w_new = (1 - r) w + r C (yi - wTxi) xi
            each element of w  is scaled by (1 - r)
            each element of xi is scaled by r C (yi - wTxi)
            after scaling them add them up and update the w :)
            """
            xi_scaled = list()
            w_scaled  = list()
            w_new     = list()

            for x in xi:
                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                factor = float(r) * float(yi -  wTxi)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #logging.debug('factor:%f',factor) 

                xi_scaled.append( factor * float(x) )
                #logging.debug(xi_scaled)

            for wt in w:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                factor = 1 - (float(r) * float(c))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #logging.debug('r     :%f',r)
                #logging.debug('c     :%f',c)
                #logging.debug('factor:%f',factor)

                w_scaled.append( float(factor) * float(wt) )
                #logging.debug(w_scaled)

            w_new = sum( [w_scaled,xi_scaled], axis=0)
            #logging.debug(w_new)

            w = w_new
            t = t + 1

        logging.debug('epoch :%d', epoch)
        logging.debug('w: %s' % ( str(w) ) )

    with open(weightsFile,'w') as f:
        for weights in w:
            f.write(str(weights) + str('\n'))
    
    logging.debug( 'training complete' )
    logging.debug( w )
    logging.debug( weightsFile )


    return    


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description = "This python program implements the logistic regression based \
                                                     classifier using stochastic gradient descent as a training algorithm " )

    parser.add_argument("-mode", help='name of the mode: like train or test',required=True)
    parser.add_argument("-data", help='name of the data file  train or test', required=True)
    parser.add_argument("-T", help='the number of epochs that will be run for sgd', required=False)
    parser.add_argument("-c", help='the value of c for use in w update rule and deciding the learning rate', required=False)
    parser.add_argument("-rou", help='the value of initial learning rate for use in w update rule', required=False)
    parser.add_argument("-wname",help='please pass the a id like prefix that distinguishes 1 w from another',required=False)

    args = parser.parse_args()

    trainingFile = args.data
    X = open(trainingFile).readlines()

    if ( args.mode == 'train_cv'):
    
        learner = '01_linregr'
        logging.debug("initiating cv..")
        K = 2

        cvResultName = str('./data/cv_results/') + str('cvr.csv')
        with open(cvResultName,'w') as f:
            header = str('learner,hyperparameters,max_error,mse_error\n')
            f.write(header)

        T_range     = [20]
        c_range     = [1,2,3,4,5,6]  
        rou_range   = [0.0001,0.0002,0.0003,0.0004]

        maxError  = list()
        meanSqError = list()
        for T in T_range:
            for c in c_range:
                for rou in rou_range:
                    logging.info( 'T: %d, c: %d, rou:%f' % ( T , c , rou ))
                    
                    learner = "01_linregr"
                    hyperparameters = 'c_' + str(c) +'_rou_'+ str(rou)
                    path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 
                    cmd = 'mkdir -p ' + str(path) 
                    os.system(cmd)

                    weightsFile = path + hyperparameters + '.w'

                    for training, validation in kFoldCrossValidation(X, K):
	                #for x in X: assert (x in training) ^ (x in validation), x
	                for x in X: next 
                        linregr_train(training,T,c,rou)
                        maxErr, meanSqErr = linregr_test(validation,T,c,rou)
                        maxError.append(maxErr)
                        meanSqError.append(meanSqErr)
        final_maxError    = max(maxError)
        final_meanSqError = max(meanSqError)         
        
        with open(cvResultName,'a') as e:
            line  = str(learner)
            line += ','
            line += str(hyperparameters)
            line += ','
            line += str(final_maxError) 
            line += ','
            line += str(final_meanSqError) + "\n"
            e.write(line)
        e.close()        
                    
        

    if ( (args.mode == 'train') or (args.mode == 'test') ):
        T                 = int(args.T)       # number of epoch
        rou               = float(args.rou)   # hyperparameter that determines the learning rate r
        c                 = float(args.c)     # this is used for scaling the w and for learning rate r
        weightsFile       = str(args.wname)
        logging.info('HyperParameters: T %d c %f rou %f',T,c,rou)
        logging.info('weightsFile: %s', weightsFile)

    if args.mode == 'train' :
        logging.info('--mode : train')
        linregr_train(X,T,c,rou)
