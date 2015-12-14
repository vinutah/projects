#!/bin/bash

echo $0
echo $1

trans='transform'
dist='distance'
cv='train_cv'
train='train'
testin='test'

cmd="python ./src/sgd.py --data"
mode="--mode $1"
k_fold_cv="--v 10"
epochs="--T 50"

c="--c 100"
rou="--rou 0.001"
weights="-wname"

data_set_0_train="./data/data0/train0.10"
data_set_0_test="./data/data0/test0.10"

data_set_1_train="./data/astro/original/train"
data_set_1_test="./data/astro/original/test"

data_set_2_train="./data/astro/original/train.transform"
data_set_2_test="./data/astro/original/test.transform"

data_set_3_train="./data/astro/scaled/train"
data_set_3_test="./data/astro/scaled/test"

data_set_4_train="./data/astro/scaled/train.transform"
data_set_4_test="./data/astro/scaled/test.transform"

weights_0="--wname data0"
weights_1="--wname astro_original"
weights_2="--wname astro_origonal_transformed"
weights_3="--wname astro_scaled"
weights_4="--wname astro_scaled_transformed"

data_set_0_c="--c 100"
data_set_0_rou="--rou 0.001"
opt_weights_0="--wname opt_data0"

data_set_1_c="--c 1"
data_set_1_rou="--rou 0.001"
opt_weights_1="--wname opt_astro_original"

data_set_2_c="--c 0.1"
data_set_2_rou="--rou 0.001"
opt_weights_2="--wname opt_astro_origonal_transformed"

data_set_3_c="--c 1000"
data_set_3_rou="--rou 0.001"
opt_weights_3="--wname opt_astro_scaled"

data_set_4_c="--c 100"
data_set_4_rou="--rou 0.001"
opt_weights_4="--wname opt_astro_scaled_transformed"

if [ $1 = $trans ]
then
    echo initiating feature transformation
    echo $cmd $data_set_1_train $mode
    echo $cmd $data_set_3_train $mode
    $cmd $data_set_1_train $mode
    $cmd $data_set_3_train $mode
fi

if [ $1 = $dist ]
then
    echo calculating the distance of the farthest data point from the origin
    echo $cmd $data_set_0_train $mode
    echo $cmd $data_set_1_train $mode
    echo $cmd $data_set_2_train $mode
    echo $cmd $data_set_3_train $mode
    echo $cmd $data_set_4_train $mode
    $cmd $data_set_0_train $mode
    $cmd $data_set_0_test $mode
    $cmd $data_set_1_train $mode
    $cmd $data_set_1_test $mode
    $cmd $data_set_2_train $mode
    $cmd $data_set_2_test $mode
    $cmd $data_set_3_train $mode
    $cmd $data_set_3_test $mode
    $cmd $data_set_4_train $mode
    $cmd $data_set_4_test $mode


fi

if [ $1 = $cv ]
then
    echo cross validation on 5 data sets
    echo $cmd  $data_set_0_train $mode $k_fold_cv $weights_0
    echo $cmd  $data_set_1_train $mode $k_fold_cv $weights_1  
    echo $cmd  $data_set_2_train $mode $k_fold_cv $weights_2
    echo $cmd  $data_set_3_train $mode $k_fold_cv $weights_3
    echo $cmd  $data_set_4_train $mode $k_fold_cv $weights_4
    $cmd  $data_set_0_train $mode $k_fold_cv $weights_0
    $cmd  $data_set_1_train $mode $k_fold_cv $weights_1  
    $cmd  $data_set_2_train $mode $k_fold_cv $weights_2
    $cmd  $data_set_3_train $mode $k_fold_cv $weights_3
    $cmd  $data_set_4_train $mode $k_fold_cv $weights_4
fi

if [ $1 = $train ]
then
    echo initiating training on 5-data sets
    echo $cmd $data_set_0_train $mode $epochs $data_set_0_c $data_set_0_rou $opt_weights_0  
    echo $cmd $data_set_1_train $mode $epochs $data_set_1_c $data_set_1_rou $opt_weights_1  
    echo $cmd $data_set_2_train $mode $epochs $data_set_2_c $data_set_2_rou $opt_weights_2 
    echo $cmd $data_set_3_train $mode $epochs $data_set_3_c $data_set_3_rou $opt_weights_3 
    echo $cmd $data_set_4_train $mode $epochs $data_set_4_c $data_set_4_rou $opt_weights_4 
    $cmd $data_set_0_train $mode $epochs $data_set_0_c $data_set_0_rou $opt_weights_0  
    $cmd $data_set_1_train $mode $epochs $data_set_1_c $data_set_1_rou $opt_weights_1  
    $cmd $data_set_2_train $mode $epochs $data_set_2_c $data_set_2_rou $opt_weights_2 
    $cmd $data_set_3_train $mode $epochs $data_set_3_c $data_set_3_rou $opt_weights_3 
    $cmd $data_set_4_train $mode $epochs $data_set_4_c $data_set_4_rou $opt_weights_4 
fi

if [ $1 = $testin ]
then
    echo initiating the use of the learned hyperplane/weights to classify test examples
    echo $cmd $data_set_0_test $mode $epochs $data_set_0_c $data_set_0_rou $opt_weights_0  
    echo $cmd $data_set_1_test $mode $epochs $data_set_1_c $data_set_1_rou $opt_weights_1  
    echo $cmd $data_set_2_test $mode $epochs $data_set_2_c $data_set_2_rou $opt_weights_2 
    echo $cmd $data_set_3_test $mode $epochs $data_set_3_c $data_set_3_rou $opt_weights_3 
    echo $cmd $data_set_4_test $mode $epochs $data_set_4_c $data_set_4_rou $opt_weights_4 
    $cmd $data_set_0_test $mode $epochs $data_set_0_c $data_set_0_rou $opt_weights_0  
    $cmd $data_set_1_test $mode $epochs $data_set_1_c $data_set_1_rou $opt_weights_1  
    $cmd $data_set_2_test $mode $epochs $data_set_2_c $data_set_2_rou $opt_weights_2 
    $cmd $data_set_3_test $mode $epochs $data_set_3_c $data_set_3_rou $opt_weights_3 
    $cmd $data_set_4_test $mode $epochs $data_set_4_c $data_set_4_rou $opt_weights_4 
fi
