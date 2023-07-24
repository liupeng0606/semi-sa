import torch
from transformers import RobertaModel, RobertaTokenizer
model_name = 'roberta-base'  # Specify the RoBERTa model name
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
def get_roberta_representation(sentence):
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    outputs = model(input_ids)
    representation = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    return representation
sentence = "This is an example sentence."
representation = get_roberta_representation(sentence)


print(representation)