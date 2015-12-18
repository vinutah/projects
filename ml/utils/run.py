#!/usr/bin/python
__author__ = 'vinu joseph'

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

#for executing system commands
import os
import sys


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


def writeError(learner, hyperparameters, mode):
    
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

"""
    if mode == "test_cv":
        path        = './data/cv_results/'
        cv_results  = path + 'cv_results' + '.csv'

        with open(cv_results,'a') as c:
            line  = str(learner)
            line += ','
            line += str(hyperparameters) 
            line += ','
            line += str(maxError)
            line += ','
            line += str(meanSqError)
            line += '\n'
            c.write(line)
        c.close()
"""


def linregr(T,c,rou,mode):
    """method for invoking the linear regression with hyperparameters"""
    logging.debug("hp: T=%d c=%f rou=%f" % ( T,c,rou ) )
    #print ("hp: T=%d c=%f rou=%f" % ( T,c,rou ) )

    learner = "01_linregr"
    hyperparameters = 'c_' + str(c) +'_rou_'+ str(rou)
    path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 
    cmd = 'mkdir -p ' + str(path) 
    os.system(cmd)

    weightsFile = path + hyperparameters + '.w'
    
    if mode == 'train' or mode == 'train_cv' :
        cmd = 'python ./learners/01_linregr/linregr.py'+\
              ' -T ' + str(T) + ' -c ' + str(c) + ' -rou ' + str(rou) + \
              ' -mode '  + str(mode) + ' -data ' + str('./data/training/train.svm') + \
              ' -wname ' + str(weightsFile)
        logging.debug( cmd )
        print '\n'
        os.system(cmd)
   
    if mode == 'test' :
        cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path) + ' -mode ' + 'ml_model ' + ' -weights ' + str(weightsFile)
        logging.debug( cmd )
        print '\n'
        os.system(cmd)
        writeError(learner, hyperparameters, mode)

    return

def liblinear(s,p,e,c,mode):
    """method for invoking the lib linear with hyperparameters"""
    logging.debug("hp: %f,%f,%f,%f" % ( s,p,e,c ) )
    #print ("hp: s=%f p=%f e=%f c=%f" % ( s,p,e,c ) )
    
    learner = "02_liblinear"

    hyperparameters = 's_' + str(s) +'_p_'+ str(p) + '_e_' + str(e) + '_c_' + str(c)
    path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 
    cmd = 'mkdir -p ' + str(path) 
    os.system(cmd)

    weightsFile = path + hyperparameters + '.m'

    if mode == 'train_cv' :
        cmd = './learners/02_liblinear/train' +\
                ' -v 6' +\
                ' -s ' + str(s) + ' -p ' + str(p) + ' -e ' + str(e) + ' -c ' + str(c) +\
                ' ./data/training/train.svm ' +\
                str(weightsFile)
        logging.debug( cmd )
        os.system(cmd)
          
 
    if mode == 'train':
        cmd = './learners/02_liblinear/train' +\
                ' -s ' + str(s) + ' -p ' + str(p) + ' -e ' + str(e) + ' -c ' + str(c) +\
                ' ./data/training/train.svm ' +\
                str(weightsFile)
        logging.debug( cmd )
        os.system(cmd)
   
    if mode == 'test':
        cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path) + ' -mode ' + 'ml_model ' + ' -weights ' + str(weightsFile)
        logging.debug( cmd )
        os.system(cmd)
        writeError(learner, hyperparameters, mode)

    return

def libsvm(s,p,e,c,g,r,t,mode):
    """method for invoking the lib svm with hyperparameters"""
    logging.debug("hp: s=%f p=%f e=%f c=%f g=%f r=%f t=%f" % ( s,p,e,c,g,r,t ) )
    
    learner = "03_libsvm"
    
    hyperparameters = 's_' + str(s) +'_p_'+ str(p) + '_e_' + str(e) + '_c_' + str(c) + '_g_' + str(g) + '_r_' + str(r) + '_t_' + str(t)
    path = './data/solutions/' + str(learner) + '/' + hyperparameters + '/' 
    cmd = 'mkdir -p ' + str(path) 
    os.system(cmd)
    
    modelFile = path + hyperparameters + '.m'
    weightsFile = path + hyperparameters + '.w'

    if mode == "train":
        cmd = './learners/03_libsvm/svm-train' +\
                ' -s ' + str(s) +' -p '+ str(p) + ' -e ' + str(e) + ' -c ' + str(c) + ' -g ' + str(g) + ' -r ' + str(r) + ' -t ' + str('t') +\
                ' ./data/training/train.svm ' +\
                str(modelFile)
        logging.debug( cmd )
        print '\n'
        os.system(cmd)
        getWeights(3,modelFile,weightsFile)
    
    if mode == "test":
        cmd = 'python ./pde/fd1d_heat_explicit_test.py ' + '-solve ' + str(path) + ' -mode ' + 'ml_model ' + ' -weights ' + str(weightsFile)
        logging.debug( cmd )
        print '\n'
        os.system(cmd)
        writeError(learner, hyperparameters, mode)

    return
    
def getWeights(fvlen,modelfname,weightsFile):
    l = 0
    rho = 0.00
    c = []
    fv = []
    fn_coef = []
    fn = ""
    svidx=0
    
    # Extract values of total_sv, rho and SV
    # from the SVM model file
    # Pass 1: Get the index of the variables
    # from the list content[]
    fmodel = open(modelfname,'r')
    content = fmodel.readlines()
    fmodel.close()
    for i in range(len(content)):
      line = content[i].replace('\n','')
      if "total_sv" in line:
        l = int(line.split()[1])
        print "total_sv = " + str(l)
      elif "rho" in line:
        rho = float(line.split()[1])
        print "rho = " + str(rho)
      elif "SV" in line:
        svidx = i+1
        print "SV = " + str(svidx)
        
    # Pass 2: Read the values of the variables
    # from the list content[] using the indices
    # obatined in pass 1
    for i in range(svidx,svidx+l):
      line = content[i].replace('\n','')
      lnfields = line.split()
      c.append(float(lnfields[0]))
      tempfv = []
      for j in range(1,1+fvlen):
        tempfv.append(float(lnfields[j].split(':')[1]))
      fv.append(tempfv)
    row = len(fv)
    col = len(fv[row-1])
    for j in range(col):
      temp_coef = 0.0
      for i in range(row):
        temp_coef = temp_coef + (c[i]*fv[i][j])
      fn_coef.append(temp_coef)

    with open(weightsFile,'w') as w:
        line  = str(fn_coef[0])
        line += '\n'
        line += str(fn_coef[1])
        line += '\n'
        line += str(fn_coef[2])
        w.write(line)
    w.close()
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

    if args.mode == 'cv':
        
        cmd = 'mkdir -p ./data/cv_results/'
        os.system(cmd)
        
        # dummy hyperparameters
        # the learner to use ranges of hyperparameters
        # and printout the cv errors 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.using == '01_linregr':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            logging.debug('invoking 10-fold cross validation for 01_linregr')
            T=0
            c=0
            rou=0            
            mode = 'train_cv'
            linregr(T,c,rou,mode)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.using == '02_liblinear':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            logging.debug('invoking 10-fold cross validation for 02_liblinear')
            mode = 'train_cv'
            S_range=[11,12,13]
            P_range=[0.001,0.01,0.1,1,10]
            E_range=[0.001,0.01,0.1,1,10]
            C_range=[0.001,0.01,0.1,1,10]

            for s in S_range:
                for p in P_range:
                    for e in E_range:
                        for c in C_range: 
                            liblinear(s,p,e,c,mode)
            
            hp_list  = list()
            cva_list = list()

            path     = './data/cv_results/'
            filename = '02_liblinear_cv_results.csv'
            filepath = path + filename

            with open(filepath,'a') as cvr:
                path_raw     = './data/cv_results/' 
                filename_raw = '02_liblinear_cv_raw.csv'
                filepath_raw = path_raw + filename_raw
                with open(filepath_raw,'r') as cv:
                    C = cv.readlines()
                    for i in range(len(C)):
                        hpKey = 'DEBUG:root:hp:'
                        if hpKey in C[i]:
                            hps = C[i].strip().split(" ")
                        acKey = 'Cross Validation Mean'
                        if acKey in C[i]:
                            er = C[i].strip().split("=")
                            line  = str(hps[1]) + str(',') + str(er[1].lstrip())
                            line += '\n'
                            cvr.write(line)
                cv.close()
            cvr.close()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.using == '03_libsvm':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            logging.debug('invoking 10-fold cross validation for 03_libsvm')

    if args.mode  == 'train':
        mode = 'train'
        if args.using == '01_linregr':
            logging.debug('invoking our linear regression to learn the solution for the bvp')
            logging.info(' L2 regularized linear regression is also called Ridge regression')
            T    = 3
            c    = 1
            rou  = 0.0001
            linregr(T,c,rou,mode)

        if args.using == '02_liblinear':
            logging.debug('invoking the LIBLINEAR to learn the solution for the bvp')
            s    = 12
            p    = 0.000001
            e    = 0.0000001
            c    = 100
            liblinear(s,p,e,c,mode)
        
        if args.using == '03_libsvm':
            logging.debug('invoking the LIBSVM liblinear to learn the solution for the bvp')
            s = 3
            p = 0.01
            e = 0.01
            c = 100
            g = 1
            r = 1
            t = 0 
            libsvm(s,p,e,c,g,r,t,mode)
            
    if args.mode  == 'test':
        mode = 'test'

        if args.using == '01_linregr':
            T    = 30
            c    = 1
            rou  = 0.0001
            linregr(T,c,rou,mode)

        if args.using == '02_liblinear':
            logging.debug('invoking the pde solver with learned model')
            s    = 12
            c    = 100
            p    = 0.000001
            e    = 0.0000001
            liblinear(s,p,e,c,mode)
            

        if args.using == '03_libsvm':
            logging.debug('invoking the pde solver with learned model')
            learner = args.using
            s = 3
            c = 100
            p = 0.01
            e = 0.01
            t = 0 
            r = 1
            g = 1
            libsvm(s,p,e,c,g,r,t,mode)
            

