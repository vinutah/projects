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
perl utils/sample.pl 0.1 ./data/training/uni_train.svm
perl utils/sample.pl 0.1 ./data/training/tri_train.svm
perl utils/sample.pl 0.1 ./data/training/pwl_train.svm


#CROSS VALIDATION
python ./utils/run.py -mode cv -using 01_linregr -bvp uni
python ./utils/run.py -mode cv -using 01_linregr -bvp tri
python ./utils/run.py -mode cv -using 01_linregr -bvp pwl

python ./utils/run.py -mode cv -using 02_liblinear -bvp uni &> ./data/cv_results/02_liblinear_cv_uni_raw.csv
python ./utils/run.py -mode cv -using 02_liblinear -bvp tri &> ./data/cv_results/02_liblinear_cv_tri_raw.csv
python ./utils/run.py -mode cv -using 02_liblinear -bvp pwl &> ./data/cv_results/02_liblinear_cv_pwl_raw.csv

python ./utils/run.py -mode cv -using 03_libsvm -bvp uni &> ./data/cv_results/03_libsvm_cv_uni_raw.csv
python ./utils/run.py -mode cv -using 03_libsvm -bvp tri &> ./data/cv_results/03_libsvm_cv_tri_raw.csv
python ./utils/run.py -mode cv -using 03_libsvm -bvp pwl &> ./data/cv_results/03_libsvm_cv_plw_raw.csv

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
