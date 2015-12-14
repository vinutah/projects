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
        if args.action== 'solve':
            logging.debug('invoking the original pde solver')
            path = './data/solutions/original/'
            cmd = 'mkdir -p ' + str(path)
            os.system(cmd)
            cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path)
            os.system(cmd)
        
"""
    if args.mode  == 'learn':

    if args.mode  == 'cv':
    if args.mode  == 'test':
    if args.mode  == 'train':

    
    if args.using == '01_sgdsvm':
    if args.using == '02_logregr':
    if args.using == '03_liblinear':
    if args.using == '04_libsvm':
"""
