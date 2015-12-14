#!/usr/bin/python
__author__ = 'vinu joseph'

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

#for executing system commands
import os

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


def writeError(learner, hyperparameters):
    
    """
    original "solution : ./data/solutions/original/h_test01.txt 
    learned  solution  : ./data/solutions/03_liblinear/s_12_c_100_p_1e-06_e_1e-07/h_test01.txt  
    """
    path        = './data/solutions/'
    output      = 'h_test01.txt'

    filename_o  = path + 'original/'  + output
    filename_ml = path + str(learner) + '/' + str(hyperparameters) + '/' + output

    print filename_o
    print filename_ml

    """ 
    compare the temperatures of the final timestep
    take MSE and write to a file as error in the same dir as solutions
    """
    
    o_list = []
    with open(filename_o) as orig:
        OT = orig.readlines()
        for i in range(len(OT)):
            o_list.append(OT[i].strip().split("  ")[-1])
    
    ml_list = []
    with open(filename_ml) as ml:
        ML = ml.readlines()
        for i in range(len(ML)):
            ml_list.append(ML[i].strip().split("  ")[-1])

    maxError, meanSqError = getErrors(o_list,ml_list)
    filename = 'errors.txt'
    errorFileName = path + str(learner) + '/' + str(hyperparameters) + '/' + filename
    
    with open(errorFileName,'w') as e:
        line = str(maxError) + str(" ") + str(meanSqError) + "\n"
        e.write(line)
    e.close()        

    return

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
        if args.using == '03_liblinear':
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

        """ calculate errors between the generated solutions """
        writeError(learner, hyperparameters)


"""
    if args.mode  == 'cv':
    if args.mode  == 'train':
"""

