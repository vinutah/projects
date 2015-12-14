#!/usr/bin/python
__author__ = 'vinu joseph'

import os
import sys
import random
import argparse
from numpy import sum
import math
import logging

logging.basicConfig(level=logging.INFO,) #format='%(asctime)s - %(levelname)s - %(message)s')

def distanceFromOrigin(X):
    xi_d = list()
    for xi in X:
        x_i = xi.strip().split(' ')
        del x_i[0]

        xi = list()
        for features in range(len(x_i)):
            xi.append(float(x_i[features].split(':')[-1]))
        
        xi_d.append(float(math.sqrt(float(sum( float(xi[i])*float(xi[i]) for i in range(len(xi)))))))

    logging.debug(xi_d)
    logging.debug('size of distance array %d',len(xi_d))    

    outfile = trainingFile + '.distance'
    with open(outfile, 'w') as f:
        logging.info('distance of the farthest point from the origin: %f \n',max(xi_d))
        f.write(str(float(max(xi_d))))
        f.write('\n') 
        f.close()    

def featureTransformation(X):
    logging.debug('in featureTransformation')
    outfile = trainingFile + '.transform'
    with open(outfile, 'w') as f:
        for xi in X:
            x_i = xi.strip().split(' ')
            y_i = x_i[0]
            del x_i[0]

            xi = list()
            for features in range(len(x_i)):
                xi.append(float(x_i[features].split(':')[-1]))
                logging.debug(xi)
            logging.debug(xi)
            logging.debug('length of feature space: %d',len(xi))
            
            for i in range(len(xi)):
                logging.debug('x%d %f:',i,xi[i])

            xi_t = list() 
            for i in range(len(xi)):
                for j in range(len(xi)):
                    if (j >= i):
                        logging.debug('x%d %f and x%d %f:',i,xi[i],j,xi[j])
                        xi_t.append(float(xi[i] * xi[j]))

            logging.debug(len(xi_t))
            f.write(y_i)
            f.write(' ')
            
            for featureIndex in range(len(xi_t)):
                f.write(str(featureIndex))
                f.write(':')
                f.write(str(xi_t[featureIndex]))
                f.write(' ')
            f.write('\n')

    f.close()
        
def kFoldCrossValidation(X,K):
    for k in xrange(K):
        training   = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def SGDClassifer_train(X,T,rou,c):
    x_1  = X[0].strip().split(' ')
    w    = [0 for i in range(len(x_1)-1)]  
    t   = 1
    for epoch in range(0,T+1):
        random.shuffle(X)
        for xi in X:
            logging.debug(xi)
            r = (rou) / (1 + (rou*(t/c)))
            x_i = xi.strip().split(' ')
            yi = x_i[0]
            del x_i[0]
            logging.debug(yi)
            xi = list()
            for features in range(len(x_i)):
                xi.append(x_i[features].split(':')[-1])
            logging.debug(xi)
            wTxi = sum( [ float(w[f]) * float(xi[f]) for f in range(len(xi)) ]  )
            logging.debug(wTxi)
            xi_scaled = list()
            w_scaled  = list()
            w_new     = list()
            logging.debug( 'w : %s', w)
            logging.debug( 'condition = %f', float(yi) * float(wTxi))
            if ((int(yi) * float(wTxi)) <= 1):
                logging.debug( "in-if")
                for x in xi:
                    factor = float(r) * float(c) * float(yi)
                    logging.debug( factor )  
                    xi_scaled.append(float(factor) * float (x))
                for wt in w:
                    w_scaled.append( ((float(1)-float(r)))  * float (wt))
                logging.debug( xi_scaled )
                logging.debug( w_scaled )
                w_new = sum([w_scaled,xi_scaled], axis=0)
                logging.debug( w_new )
            else:
                logging.debug( "in-else" )
                for wt in w:
                    w_scaled.append( ((float(1)-float(r)))  * float (wt))
                w_new = w_scaled 
            w = w_new
            t = t + 1
            logging.debug( 'epoch %d : example %d' % (epoch,t) ) 
    
    os.system('mkdir -p ./data/weights/')

    with open(weightsFile, 'w') as f:
        for weights in w:
            f.write(str(weights) + str('\n'))
    
    logging.debug( 'training complete' )
    logging.debug( w )

def SGDClassifer_test(X,T,rou,c):
    accNr = 0
    with open(weightsFile, 'r') as f:
        w = [line.strip() for line in f]
    logging.debug( w )
     
    normOfW = float(math.sqrt(float(sum( float(w[i])*float(w[i]) for i in range(len(w))))))
    logging.info( "norm of the weight vector : %f",normOfW )
    
    distOnCorrectSide = list()
    distOnWrongSide   = list()
    
    for xi in X:
        x_i = xi.strip().split(' ')
        yi = x_i[0]
        del x_i[0]

        xi = list()
        for features in range(len(x_i)):
            xi.append(x_i[features].split(':')[-1])
        logging.debug( xi )

        wTxi = sum( [ float(w[f]) * float(xi[f]) for f in range(len(xi)) ]  )
        logging.debug( wTxi )

        distFromHyperPlane    = float(wTxi) / float(normOfW)
        yi_distFromHyperPlane = int(yi) * distFromHyperPlane        
 
        if ( yi_distFromHyperPlane >= 0 ):
            distOnCorrectSide.append( float(int(yi)*(wTxi/normOfW)) )
        else:
            distOnWrongSide.append( -1 * float(int(yi)*(wTxi/normOfW)) )

        if wTxi >= 0:
            prediction = +1
        else:
            prediction = -1

        logging.debug( 'actual : %s predicted : %s' % ( yi , prediction) )
    
        if (int(yi) == int(prediction)):
            accNr = accNr + 1
            #logging.debug( 'yes :%d' % (accNr)
    acc = float(accNr)*100/len(X)

    if ( len(distOnCorrectSide) !=0 ):
        marginWrtCorrectSide = min(distOnCorrectSide)
        logging.info( 'Margin-CORRECT_SIDE %f', marginWrtCorrectSide )
    
    if ( len(distOnWrongSide) !=0 ): 
        marginWrtWrongSide = min(distOnWrongSide)
        logging.info( 'Margin-INCORRECT_SIDE %f', marginWrtWrongSide )
    
    logging.debug( 'testing complete')
    logging.debug( 'correct  :', accNr)
    logging.debug( 'total    :', len(X))
    logging.debug( 'accuracy_tmp :', acc)
    return acc
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='This program implements the SVM learner \
    using stochastic sub gradient descent as a training algorithm using python')

    parser.add_argument("-mode","--mode",     help='training mode',           required=True)
    parser.add_argument("-data","--data",     help='data file in svm format', required=True)
    parser.add_argument("-wname","--wname",   help='prefix to the generated weights file', required=False)
    
    parser.add_argument("-v","--v", type=int, help='k-fold cross validation', required=False)
    parser.add_argument("-c","--c", help='value of c, the knob to control the weightage of regularizer vs loss', required=False)
    parser.add_argument("-T","--T", help='the number of epochs to run the sdg ', required=False)
    parser.add_argument("-rou","--rou",help='numerator for chooseing learning rate r, hyperparameter', required=False)

    parser.add_argument("-distance","--distance", type=int, help='calculate distace of the feature vectors from Origin', required=False)
    
    args = parser.parse_args()

    trainingFile       = args.data
    weightsFilePrefix  = args.wname
     
    X    = open(trainingFile).readlines()
    logging.info( 'trainingData      : %s',    trainingFile)
    logging.info( 'examples          : %d',len(X)) 
    logging.info( 'weightsFilePrefix : %s',weightsFilePrefix)

    if args.mode == 'transform':
        logging.info('--mode : transform')
        featureTransformation(X)  

    if args.mode == 'distance':
        logging.info('--mode : distance')
        distanceFromOrigin(X)
    
    if args.mode == 'train_cv':
        K = args.v 
        logging.info( 'initiating %d-fold cross-validation', K)
      
        os.system('mkdir -p ./data/cv_results') 
        cvResultName = str('./data/cv_results/') + str(weightsFilePrefix) + str('_cv.csv')
      
        with open(cvResultName,'w') as f:
            header    = str('T,rou,c,CV_accuracy\n')
            f.write(header)

        T_range   = [50]
        rou_range = [0.001,0.01,0.1,1,10,100,1000]
        c_range   = [0.001,0.01,0.1,1,10,100,1000]
        
        for T in T_range:
            for rou in rou_range:
                for c in c_range:
                    logging.info( 'T=%d , rou=%f, c=%f' % ( T , rou , c) )

                    weightsFile =   str('./data/weights/') + 'weights_'\
                                  + str(weightsFilePrefix)\
                                  + '_T' + str(T)\
                                  + '_rou' + str(rou)\
                                  + '_c' +str(c) 
                        
                    acc = 0
                    for training, validation in kFoldCrossValidation(X, K):
	                #for x in X: assert (x in training) ^ (x in validation), x
	                for x in X: next 
                        SGDClassifer_train(training,T,rou,c)
                        acc += SGDClassifer_test(validation,T,rou,c)
                                
                    acc_cv = acc/K
                    logging.info( 'cross validaiton accuracy %f:', acc_cv )

                    with open(cvResultName,'a') as f:
                        cv_result = str(T) + str(',') + str(rou) + str(',') + str(c) + str(',') + str(acc_cv) + str(',') + str('\n')
                        f.write(cv_result)
    
    if ( (args.mode == 'train') or (args.mode == 'test') ):
        T   = int(args.T)
        rou = float(args.rou)
        c   = float(args.c)
        logging.info('HyperParameters: T %s rou %s c %s',T,rou,c)

        weightsFile =   str('./data/weights/')\
                         + 'weights_'\
                         + str(weightsFilePrefix)\
                         + '_T' + str(T)\
                         + '_rou' + str(rou)\
                         + '_c' +str(c) 

    if args.mode == 'train':
        logging.info('--mode : train')
        SGDClassifer_train(X,T,rou,c)
    
    if args.mode == 'test':
        logging.info('--mode : test')
        acc = SGDClassifer_test(X,T,rou,c)
        logging.info('Accuracy : %f \n\n', acc)
