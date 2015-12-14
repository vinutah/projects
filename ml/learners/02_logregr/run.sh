#!/bin/bash

echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
echo Running the logistic regression classifier on ASTRO-ORIGINAL
python ./src/logisticRegr/lrClassifier.py --data ./data/astro/original/test --T 80 --rou 0.001 --s 100000 --wname optimal_astro_original --mode test

echo Running the logistic regression classifier on ASTRO-SCALED
python ./src/logisticRegr/lrClassifier.py --data ./data/astro/scaled/test --T 80 --rou 0.001 --s 100000 --wname optimal_astro_scaled --mode test
echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
