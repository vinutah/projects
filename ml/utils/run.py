#!/usr/bin/python
__author__ = 'vinu joseph'

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

#for executing system commands
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = "this program defines the user interface of the project\
                                      for the class cs635 machine learning of fall 2015, help pick the best\
                                      learner and associated settings for a give pde as a boundary value problem")

    parser.add_argument("-mode",   help='name of the mode    ',required=True)
    parser.add_argument("-action", help='name of the action to be performed',required=False)
    
    parser.add_argument("-using",  help='name of the learner ',required=False)
    args = parser.parse_args()

    if args.mode  == 'original':
        if args.action == 'solve':
            logging.debug('invoking the original pde solver')
            path = './data/solutions/original/'
            cmd = 'mkdir -p ' + str(path)
            os.system(cmd)
            cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path) + ' -mode ' + 'native'
            os.system(cmd)

    if args.mode  == 'learn':
        if args.using == '03_liblinear':
            logging.debug('invoking the LIBLINEAR to learn the solution for the bvp')
            c    = 100
            p    = 0.000001
            e    = 0.0000001
            s    = 12
            learner = args.using

            hyperparameters = 's_' + str(s) +'_c_'+ str(c) + '_p_' + str(p) + '_e_' + str(e)
            path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 
            cmd = 'mkdir -p ' + str(path) 
            os.system(cmd)

            weightsFile = path + hyperparameters + '.m'
            cmd = './learners/03_liblinear/train' +\
                    ' -s ' + str(s) + ' -c ' + str(c) + ' -p ' + str(p) + ' -e ' + str(e) +\
                    ' ./data/training/train.svm ' +\
                    str(weightsFile)
            print cmd
            os.system(cmd)
        
        if args.using == '01_sgdsvm':
            logging.debug('invoking the HW4_SVM to learn the solution for the bvp')
        
        if args.using == '02_logregr':
            logging.debug('invoking the HW5_LOGISTIC_REGR to learn the solution for the bvp')
        
        if args.using == '04_libsvm':
            logging.debug('invoking the LIBSVM liblinear to learn the solution for the bvp')

    if args.mode  == 'test':
        logging.debug('invoking the pde solver with learned model')
        
        c    = 100
        p    = 0.000001
        e    = 0.0000001
        s    = 12
        learner = args.using
        
        hyperparameters = 's_' + str(s) +'_c_'+ str(c) + '_p_' + str(p) + '_e_' + str(e)
        path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 

        weightsFile = path + hyperparameters + '.m'
        
        cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path) + ' -mode ' + 'ml_model ' + ' -weights ' + str(weightsFile)
        print cmd
        os.system(cmd)
"""
    if args.mode  == 'cv':
    if args.mode  == 'train':
"""

