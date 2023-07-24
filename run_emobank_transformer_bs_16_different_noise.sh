#!/usr/bin/bash



for task in emobank
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
        python model_emobank.py $task $roi $seed transformer 2000 16 $noise emobank_transformer_bs_16_different_noise
      done
    done
  done
done

# python main.py 
# for task in ["cvat", "fb", "emobank"]:
#     print("begin task: " + str(task))
#     for roi in [0.1,0.2,0.4]: 
#         print("labeled rot :"+str(roi))      
#         for i in range(5):
#             setup_seed(i)
#             train(epochs=2000, task=task, labeled_ratios=roi, submodel=BiLSTM())
#             for key in all_list:
#                 print(all_list[key])
#             print()
#     print("end task: " + str(task))
#     print()


# print()
# print("change submmodel to Transformer !")
# for task in ["cvat", "fb", "emobank"]:
#     print("begin task: " + str(task))
#     for roi in [0.1,0.2,0.4]: 
#         print("labeled rot :"+str(roi))
#         for i in range(5):
#             setup_seed(i)
#             train(epochs=2000, task=task, labeled_ratios=roi, submodel=BatchFormer())
#             for key in all_list:
#                 print(all_list[key])
#             print()
#     print("end task: " + str(task))
#     print()