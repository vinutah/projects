#!/bin/bash

#CREATE TRAINING AND TESTING DATA
#(uniform)
python ./utils/run.py -mode original_uni_1 -action solve
python ./utils/run.py -mode original_uni_2 -action solve
python ./utils/run.py -mode original_uni_3 -action solve
python ./utils/run.py -mode original_uni_4 -action solve
 
#(triangular)
python ./utils/run.py -mode original_tri_1 -action solve
python ./utils/run.py -mode original_tri_2 -action solve
python ./utils/run.py -mode original_tri_3 -action solve
python ./utils/run.py -mode original_tri_4 -action solve

#(piece wise linear)
python ./utils/run.py -mode original_pwl_1 -action solve
python ./utils/run.py -mode original_pwl_2 -action solve
python ./utils/run.py -mode original_pwl_3 -action solve
python ./utils/run.py -mode original_pwl_4 -action solve

#SAMPLING
#perl utils/sample.pl 0.1
#
#CROSS VALIDATION
#python ./utils/run.py -mode cv -using 01_linregr
#python ./utils/run.py -mode cv -using 02_liblinear &> ./data/cv_results/02_liblinear_cv_raw.csv
#python ./utils/run.py -mode cv -using 03_libsvm &> ./data/cv_results/03_libsvm_cv_raw.csv
#
#TRAINING PHASE
#Using the Best algorithm and its associated hyperparameters, retrain:
#python ./utils/run.py -mode train -using 01_linregr
#python ./utils/run.py -mode train -using 02_liblinear
#python ./utils/run.py -mode train -using 03_libsvm
#
#TESTING PHASE
#Use the weights learned above and report max and mean square Errors.
#python ./utils/run.py -mode test -using 01_linregr
#python ./utils/run.py -mode test -using 02_liblinear
#python ./utils/run.py -mode test -using 03_libsvm
