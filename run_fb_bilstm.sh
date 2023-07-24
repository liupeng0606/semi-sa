#!/usr/bin/bash



for task in fb
do 
  echo begin task: $task
  for roi in 0.4
  do
    echo  "$task -> $roi"
    for bs in 8
    do
      echo "$task -> $roi -> $bs"
      for seed in 0 100 1000 10000 100000
      do
        echo "$task -> $roi -> $bs ->  $seed"
        python model_fb.py $task $roi $seed lstm 2000 $bs 0.05 fb_bilstm
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