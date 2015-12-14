#!/usr/bin/python
__author__ = "vinu joseph"

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO,)

def generateDataSet(fv):
    for i in range(128):
        bs = "{0:07b}".format(i)
        y  = int(bs[0]) + int(bs[1]) + int(bs[2]) + int(bs[3]) + int(bs[4]) + int(bs[5]) + int(bs[6])
        #print y
        if y >= 3:
            label = 1
        else:
            label = 0    
        print ("%d 1:%s 2:%s 3:%s 4:%s 5:%s 6:%s 7:%s" % (int(label),bs[0],bs[1],bs[2],bs[3],bs[4],bs[5],bs[6]))

def naiveBayesTrain(X):
    numOfExamples = len(X)
    print 'numOfExamples : ', numOfExamples
   
    x1_yis1 = [] 
    x1_yis0 = [] 

    for j in range(1,8):
        for i in range(numOfExamples):
            e  = X[i].strip().split(' ')
            label = e[0]
            #print label
  
            if label == '1':
                f1_yis1 = e[j].split(':')[1]
                x1_yis1.append(f1_yis1) 
            
            if label == '0':
                f1_yis0 = e[j].split(':')[1]
                x1_yis0.append(f1_yis0) 
            
        
        pOfx1is1_yis1 =  float(x1_yis1.count('1')) / len(x1_yis1)
        pOfx1is0_yis1 =  float(x1_yis1.count('0')) / len(x1_yis1)
        pOfyis1       =  float(len(x1_yis1))/len(X)    

        pOfx1is1_yis0 =  float(x1_yis0.count('1')) / len(x1_yis0)
        pOfx1is0_yis0 =  float(x1_yis0.count('0')) / len(x1_yis0)
        pOfyis0       =  float(len(x1_yis0))/len(X)    
        
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "count of examples when y=1 :", len(x1_yis1)
        print "count of x1=1     when y=1 :", x1_yis1.count('1')
        print "count of x1=0     when y=1 :", x1_yis1.count('0')
        print "     p(x1=1|y=1)           :", pOfx1is1_yis1 
        print "     p(x1=0|y=1)           :", pOfx1is0_yis1 
        print "     p(y=1)                :", pOfyis1
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "count of examples when y=0 :", len(x1_yis0)
        print "count of x1=1     when y=0 :", x1_yis0.count('1')
        print "count of x1=0     when y=0 :", x1_yis0.count('0')
        print "     p(x1=1|y=0)           :", pOfx1is1_yis0 
        print "     p(x1=0|y=0)           :", pOfx1is0_yis0 
        print "     p(y=0)                :", pOfyis0
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        
        x1_yis1 = [] 
        x1_yis0 = [] 
    

def naiveBayesClassifier(X):
    numOfExamples = len(X)
    
    prior_yis1 = float(99)/128
    prior_yis0 = 1 - float(prior_yis1)

    likelihood_xi_1_y_1 = float(57)/99
    likelihood_xi_0_y_1 = float(1 - likelihood_xi_1_y_1)

    likelihood_xi_1_y_0 = float(7)/99
    likelihood_xi_0_y_0 = float(1 - likelihood_xi_1_y_0)

    print 'prior      p(y=1)      :',prior_yis1
    print 'prior      p(y=0)      :',prior_yis0

    print 'likelihood p(xi=1|y=1) :',likelihood_xi_1_y_1
    print 'likelihood p(xi=0|y=1) :',likelihood_xi_0_y_1
    print 'likelihood p(xi=1|y=0) :',likelihood_xi_1_y_0
    print 'likelihood p(xi=0|y=0) :',likelihood_xi_0_y_0
    
    error = 0
    for i in range(numOfExamples):
        e = X[i].strip().split(' ')
        label = e[0]
        
        x = []    
        for j in range(1, 8):
            x.append(e[j].split(':')[1])
        #print x
        
        product_yis1 = float(prior_yis1)
        for j in range(0, 7):
            if x[j] == '1':
                product_yis1 = product_yis1 * likelihood_xi_1_y_1
            if x[j] == '0':
                product_yis1 = product_yis1 * likelihood_xi_0_y_1
        #print 'p(y=1|x)   :', product_yis1

        product_yis0 = float(prior_yis0)
        for j in range(0, 7):
            if x[j] == '1':
                product_yis0 = product_yis0 * likelihood_xi_1_y_0
            if x[j] == '0':
                product_yis0 = product_yis0 * likelihood_xi_0_y_0
        #print 'p(y=0|x)   :', product_yis0

        if product_yis1 >= product_yis0:
            prediction = '1'
        else: 
            prediction = '0'

        if prediction != label:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' 
            print x    
            print 'p(y=1|x)   :', product_yis1
            print 'p(y=0|x)   :', product_yis0
            print 'prediction :', prediction
            print 'label      :', label
            error += 1

    err = float(error) / len(X)
    accuracy = 100 - err*100

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'total examples        :' , len(X) 
    print 'correct   predictions :' , len(X) - error
    print 'incorrent predictions :' , error
    print 'accuracy              :' , accuracy
  
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program implements the Naive Bayes \
    classifier')  

    parser.add_argument("--gen",   help='generate the data set of boolean features', required=False) 
    parser.add_argument("--train", help='MAP learning! train on the data set of boolean features', required=False) 
    parser.add_argument("--test", help='MAP learning! train on the data set of boolean features', required=False) 

    args = parser.parse_args()

    if args.gen:
        fv = int(args.gen )
        generateDataSet(fv)

    if args.train:
        svmFile = args.train
        
        X    = open(svmFile).readlines()
        logging.info( 'trainingData      : %s',svmFile)
        logging.info( 'examples          : %d',len(X)) 

        naiveBayesTrain(X)

    if args.test:
        svmFile = args.test
        
        X    = open(svmFile).readlines()
        logging.info( 'trainingData      : %s',svmFile)
        logging.info( 'examples          : %d',len(X)) 

        naiveBayesClassifier(X)
        
