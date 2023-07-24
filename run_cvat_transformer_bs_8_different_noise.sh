#!/usr/bin/bash



for task in cvat
do 
  echo begin task: $task
  for roi in 0.1 0.2 0.4
  do
    echo  "$task -> $roi"
    for noise in 0.01 0.02 0.05 0.1 0.2 0.5 
    do
      echo "$task -> $roi -> $bs"
      for seed in 0 100 1000 10000 100000
      do
        echo "$task -> $roi -> $seed ->  $noise"
        python model_cvat.py $task $roi $seed transformer 2000 8 $noise cvat_transformer_bs_8_different_noise
      done
    done
  done
done

