import pickle
import pandas as pd
import numpy as np
import torch
import random

file_path = "LLMpredictorData.pkl"
with open(file_path, 'rb') as file:
    all_data_list = pickle.load(file)
    
import kagglehub    
    
VARIANT = '9b-it'

CONFIG =  '9b'
import os
VARIANT = '9b-it'
weights_dir = kagglehub.model_download("google/gemma-2/pyTorch/gemma-2-9b-it/1") 


import gemma_config
import gemma_model_og
import gemma_tokenizer

tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# print("loaded tokenizer and checkpoint")


import sentencepiece as spm
vocab = spm.SentencePieceProcessor()
assert vocab.load(tokenizer_path)

model_config = gemma_config.get_model_config(CONFIG)
model_config.tokenizer = tokenizer_path
# model_config.quant = 'quant' in VARIANT


torch.set_default_dtype(model_config.get_dtype())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gemma_model_og.GemmaForCausalLM(model_config)


#model.load_weights(weights_dir)
# model = model.to(device).eval()

from openai import OpenAI
score = 0
explanation_dict = {}
for i in range(len(all_data_list)):
    cur_data = all_data_list[i]
    ground_truth_list = cur_data['ref_source']
    prompt = 'Given this following paper: "'
    prompt += cur_data['full_text'][:400]
    prompt += '"'
    prompt += '. \n Below is a list of its references titles and the context of each one: \n'
    
    ground_truth_title_list = []
    for j in ground_truth_list:
        ground_truth_title_list.append({value: key for key, value in all_data_list[i]['citations']['mapping'].items()}[j])
    
    for j in ground_truth_list:
        cur_title = {value: key for key, value in all_data_list[i]['citations']['mapping'].items()}[j]
        prompt += f'"{cur_title}"'
        try:
            context = cur_data['citations'][cur_title]['context']
            context_count = len(context)
            prompt += f', appearing in the following {context_count} contexts: \n'
            prompt += str(context)
        except:
            context = None
        # print(context)
        
    
    
    for k, v in cur_data['citations'].items():
        if len(k)>10 and k.lower() not in ground_truth_title_list:
            prompt += k
            context_count = len(v['context'])
            prompt += f', appearing in the following {context_count} contexts: \n'
            prompt += str(v['context'])
            prompt += '\n'

    prompt += "Among all the references of this paper, they can be ranked according to their importance to this paper. \n"
    ref_count = len(cur_data['citations'].keys())-1
    
    
    paper_title = {value: key for key, value in all_data_list[i]['citations']['mapping'].items()}[ground_truth_list[0]]
    prompt += 'Give a concise explanation (around 50 words) of which reference is the most important and why. '
    # prompt += f'By analyzing the surrounding context, give the importance rank of paper "{paper_title}" as integer number between 0 (meaning most important) and {ref_count-1} (meaning least important). '
    # prompt += f'Only give the number as your answer, DO NOT provide any reasoning. '
    prompt += 'Answer:'

    client = OpenAI(api_key = "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")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an experienced researcher."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.1
    )

    response = completion.choices[0].message.content
    explanation_dict[cur_data['id']] = response

pickle_file = "explanations.pkl"

# Saving the object as a pickle file
with open(pickle_file, 'wb') as file:
    pickle.dump(explanation_dict, file)
#     try:
#         answer = int(response)
#     except ValueError:
#         answer = ref_count
#     score += 1-answer/ref_count
    
# print(score)
# print(len(all_data_list))