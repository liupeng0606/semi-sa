import pickle as pl
import pandas as pd
import json

# data = []

# with open("./data/emobank.json", "r", encoding="utf-8") as f:
#     content = json.load(f)
#     i = 0
#     for item in content:
#         data.append([" ".join(item[0]), item[-1][0], item[-1][1]])

# pd.DataFrame(data).to_csv("./data/wu_emobank_all.csv", index = None, sep="\t", header=None)


with open("./data/emobank.pkl", "rb") as f:
    pkl_0 = pl.load(f)

    print(len(pkl_0[0][0]))

# with open("./data/processed_fb.pkl", "rb") as f:
#     pkl_1 = pl.load(f)   


# print(pkl_0[0])
# print(pkl_0[1])

# data = pd.read_csv("./data/fb.csv", usecols=[1,3]).values

# def fun(x):
#     return [x[0], x[1]]

# data = list(map(fun, data))

# processed_fb_new = list(zip(pkl_data, data))


# with open("./data/processed_fb_new.pkl", "wb") as f:
#     pl.dump(processed_fb_new, f)




# with open("./data/processed_emobank.pkl", "wb") as f:
#     pl.dump(processed_emobank, f)

# with open("./data/processed_emobank.pkl", "rb") as f:
#     processed_emobank = pl.load(f)

# print((processed_emobank)[0])



# with open("./data/processed_fb.pkl", "rb") as f:
#     processed_fb = pl.load(f)

# print((processed_fb)[0])
 


# with open("./data/processed_cvat.pkl", "rb") as f:
#     processed_cvat = pl.load(f)

# print(processed_cvat[0])
 

