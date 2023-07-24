

import torch
from transformers import AlbertModel, AlbertTokenizer
model_name = 'albert-base-v2'  # Specify the ALBERT model name
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertModel.from_pretrained(model_name).cuda()
def get_albert_representation(sentence):
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0).cuda()  # Add batch dimension
    outputs = model(input_ids)
    representation = outputs.last_hidden_state[:, 0, :].squeeze(0)  # Remove batch dimension
    return representation



import pandas as pd
data = pd.read_csv("./emobank.csv", header=None, usecols=[2,3,5]).values
all_data = []
i=0
for item in data:
    i=i+1
    sentence = item[-1]
    representation = get_albert_representation(sentence).cpu().tolist()
    all_data.append([representation, [item[0], item[1]]])
    print(i)
import numpy as np
import pickle as pl
with open("./albert/emobank_albert.pkl", "wb") as f:
    pl.dump(all_data, f)
