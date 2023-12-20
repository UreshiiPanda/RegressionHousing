#!/bin/bash

# find the 5 best dev results when tuning alpha between 0 and 10 

echo "alpha,res" > ridge_res.csv

# run ridge regression with different alphas
for alpha in $(seq 0 .1 10); do
    # make sure to setup ridge.py to only output the RMSLE on dev 
    # get dev results for each alpha and store them in a csv
    res=$(python3 ridge.py "$alpha") 
    echo "$alpha","$res" >> ridge_res.csv
done


printf "\n"
echo "Five best RMSLE's on dev:"
python3 ridge_best.py
printf "\n"


# get the best alpha 
output=$(python ridge_best.py)
best_alpha=$(echo "$output" | tail -n 1)

# print the alpha with the best results on dev
echo "Best Alpha: $best_alpha"
printf "\n"

# run Ridge with the best alpha, output goes to ridge_out.csv
python3 ridge.py $best_alpha


