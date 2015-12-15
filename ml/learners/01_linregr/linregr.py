#!/usr/bin/python
__author__ = 'vinu joseph'

import logging
logging.basicConfig(level=logging.INFO)
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

def printNegativeLL(epoch,X,w,s):
    objectiveFunc = 0.0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    wTw  = float(sum( float(w[i])*float(w[i]) for i in range(len(w))))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for xi in X:
        
        """ read the line into this list and split on spaces """
        x_i = xi.strip().split(' ')
        
        """ get the label of this exmaple and delete it from the x_i list :) """
        yi = float(x_i[0])
        del x_i[0]
        
        """ load the features into a list data structure """
        xi = list()
        for features in range(len(x_i)):
            xi.append(x_i[features].split(':')[-1])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        wTxi = float(sum( [ float(w[f]) * float(xi[f]) for f in range(len(xi)) ]  ) )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #logging.debug('w: %s yi:%d xi:%s' % ( str(w), int(yi), str(xi) ) )
        #logging.debug('wTxi : %f' % (wTxi))
        #logging.debug('wTw  : %f' % (wTw )) 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        objectiveFunc = objectiveFunc + float ( math.log ( 1 + math.exp(-1*yi*wTxi)) + (1/s*s)*wTw )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
        """ write these values in a file 
            later this will used to plot
            the negatuve log likelihood"""

    cmd = 'mkdir -p ./data/objective'
    os.system(cmd)

    filename = './data/objective/negativeLL_' + str(weightsFilePrefix) + '.data'
    
    with open(filename,'a') as f:
        f.write(str(epoch))
        f.write(str(' '))
        f.write(str(objectiveFunc))
        f.write('\n')
    f.close()
    
    return
 
def plotobjfn():
    x = list()
    y = list()
    plotFileName = './data/objective/negativeLL_' + str(weightsFilePrefix) + '.data'
    for line in open(plotFileName):
        values = line.strip().split()
        x.append(values[0])
        y.append(float(values[1]))

    xaxis = np.array(x)
    yaxis = np.array(y)
    plt.plot(xaxis,yaxis) 

    plt.show()       

    return
       
def kFoldCrossValidation(X,K):
    for k in xrange(K):
        training   = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def lrClassifier_train(X,T,s,rou):
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
        r(t, rou) = rou / (1 + rou*t/sigma*sigma)
        here the rou and sigma are hyperparameters for this algorithm
    """
    t = 0

    for epoch in range(0,T+1):
        if args.mode=='train':
            printNegativeLL(epoch,X,w,s) 
        random.shuffle(X)
        for xi in X:
            """ calculate the learning rate """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            r = (rou) / (1 + (rou*(t/s*s)))            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
            """ read the line into this list and split on spaces """
            x_i = xi.strip().split(' ')
            
            """ get the label of this exmaple and delete it from the x_i list :) """
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
            w_new = (1 - r/s*s) w + r ( 1/[1 + e- power- (yi.wTxi)] ) yi xi
            each element of w  is scaled by (1 - r/s*s)
            each element of xi is scaled by (r*yi) / (1 + e power (yi * wT * xi))
            after scaling them add them up and update the w :)
            """
            xi_scaled = list()
            w_scaled  = list()
            w_new     = list()

            for x in xi:
                #logging.debug('r     :%f',r )
                #logging.debug('yi    :%f',yi)
                #logging.debug('wTxi  :%f',wTxi)
                #logging.debug('C     :%f',(1/(1 + math.exp(yi* wTxi))) )
                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                factor = float(r) * (float(1)/float(1 + math.exp(yi * wTxi))) * float(yi)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #logging.debug('factor:%f',factor) 

                xi_scaled.append( factor * float(x) )
                #logging.debug(xi_scaled)

            for wt in w:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                factor = 1 - (float(2*r)/(s*s))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #logging.debug('r     :%f',r)
                #logging.debug('s     :%f',s)
                #logging.debug('s*s   :%f',s*s)
                #logging.debug('factor:%f',factor)

                w_scaled.append( float(factor) * float(wt) )
                #logging.debug(w_scaled)

            w_new = sum( [w_scaled,xi_scaled], axis=0)
            #logging.debug(w_new)

            w = w_new
            t = t + 1

        logging.debug('epoch :%d', epoch)
        logging.debug('w: %s' % ( str(w) ) )

    cmd = 'mkdir -p ./data/weights'
    os.system(cmd)
    
    with open(weightsFile,'w') as f:
        for weights in w:
            f.write(str(weights) + str('\n'))
    
    logging.debug( 'training complete' )
    logging.debug( w )


    return    

def lrClassifier_test(X,T,s,r):
   
    """
    open weights file in read mode and load into a list data structure 
    """ 

    accNr = 0
    with open(weightsFile,'r') as f:
        w = [line.strip() for line in f]
    logging.debug(w)

    """
    read the test file with care
    """
    for xi in X:
        x_i = xi.strip().split(' ')
        yi  = x_i[0]
        del x_i[0]

        xi = list()
        for features in range(len(x_i)):
            xi.append(x_i[features].split(':')[-1])
        #logging.debug( xi )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        wTxi = sum( [ float(w[f]) * float(xi[f]) for f in range(len(xi)) ] )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #logging.debug( wTxi )

        if (wTxi >= 0):
            prediction = +1
        else:
            prediction = -1

        #logging.debug( 'actual : %s predicted : %s' % ( yi , prediction))

        if (int(yi) == int(prediction)):
            accNr += 1

    acc = float(accNr)*100/len(X)
    logging.debug( 'testing complete')
    logging.debug( 'correct         :%d', accNr)
    logging.debug( 'total           :%d', len(X))
    logging.debug( 'accuracy        :%f', acc)
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description = "This python program implements the logistic regression based \
                                                     classifier using stochastic gradient descent as a training algorithm " )

    parser.add_argument("--mode", help='name of the mode:    like cv or train or test',required=True)
    parser.add_argument("--v", help='number of folds for cross validation         ', required=False)
    parser.add_argument("--data", help='name of the data file for cv or train or test', required=True)
    parser.add_argument("--T", help='the number of epochs that will be run for sgd', required=False)
    parser.add_argument("--s", help='the value of     sigma     for use in w update rule for LR', required=False)
    parser.add_argument("--rou", help='the value of learning rate for use in w update rule for LR', required=False)
    parser.add_argument("--wname",help='please pass the a id like prefix that distinguishes 1 w from another',required=False)

    args = parser.parse_args()

    trainingFile = args.data
    X = open(trainingFile).readlines()
    weightsFilePrefix = args.wname

    if ( (args.mode == 'train') or (args.mode == 'test') ):
        T       = int(args.T)       # number of epoch
        rou     = float(args.rou)   # hyperparameter that determines the learning rate r
        sigma   = int(args.s)       # this is used for scaling the w and for learning rate r
        logging.info('HyperParameters: T %d s %f rou %f',T,sigma,rou)

        weightsFile =   str('./data/weights/')\
                         + 'weights_'\
                         + str(weightsFilePrefix)\
                         + '_T' + str(T)\
                         + '_s' + str(sigma)\
                         + '_rou' +str(rou) 

    if args.mode == 'train':
        logging.info('--mode : train')
        lrClassifier_train(X,T,sigma,rou)
    
    if args.mode == 'test':
        logging.info('--mode : test')
        
        acc = lrClassifier_test(X,T,sigma,rou)
        logging.info('accuracy : %f \n\n', acc)

    if args.mode == 'plot':
        logging.info('--mode : plotting objective function')
        plotobjfn()

    if args.mode == 'cv':
        K = int(args.v)
        logging.info('initiating %d-fold cross validation',K)
        
        cmd = 'mkdir -p ./data/cv_results/'
        os.system(cmd)
        
        cvResultName = str('./data/cv_results/') + str(weightsFilePrefix) + str('_cv.csv')
        with open(cvResultName,'w') as f:
            header = str('Epochs,sigma,learningRate,cvAccuracy\n')
            f.write(header)

        T_range     = [80]
        s_range     = [100,    1000,  10000,  100000]
        rou_range   = [0.001,0.0001,0.00001,0.000001]

        for T in T_range:
            for s in s_range:
                for rou in rou_range:
                    logging.info( 'T: %d, s: %d, rou:%f' % ( T , s , rou ))
                    
                    weightsFile =   str('./data/weights/') + 'weights_'\
                                  + str(weightsFilePrefix)\
                                  + '_T'   + str(T)\
                                  + '_s'   + str(s)\
                                  + '_rou' + str(rou) 

                    logging.debug('weightsFIle = %s' % (weightsFile))    

                    acc = 0
                    for training, validation in kFoldCrossValidation(X, K):
	                #for x in X: assert (x in training) ^ (x in validation), x
	                for x in X: next 
                        lrClassifier_train(training,T,s,rou)
                        acc += lrClassifier_test(validation,T,s,rou)

                    acc_cv = acc/K
                    logging.info( 'cross validaiton accuracy %f:', acc_cv )

                    with open(cvResultName,'a') as f:
                        cv_result = str(T) + str(',') + str(s) + str(',') + str(rou) + str(',') + str(acc_cv) + str(',') + str('\n')
                        f.write(cv_result)
