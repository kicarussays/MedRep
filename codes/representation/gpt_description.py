"""
    Yield descriptionss for all concept IDs using ChatGPT-4o-mini.
    Input:
        1. data/concepts/tmp/[Condition, Drug, Measurement, Procedure]/final_concepts_chunk_[0-19].csv
    Output:
        1. usedata/descriptions/[Condition, Drug, Measurement, Procedure]/concept_description_[0-19].csv
        
"""

import os
import pandas as pd
import numpy as np
import openai
import argparse
from tqdm import tqdm
import gc
import time
import sys
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chunk", type=int) 
args = parser.parse_args()

openai.api_key = ""

PromptFor = {
    'Condition': '### Instruction: Briefly explain the clinical background and regarding treatments of each concept name (condition) with less than 5 sentences. Do not include sentences that are too ordinary (such as "further details would depend on the specific situation) and focus on describing the representative clinical features of the concept.',
    'Drug': '### Instruction: Briefly explain the clinical background and purpose of each concept name (drug) with less than 5 sentences. Do not include sentences that are too ordinary (such as "further details would depend on the specific situation) and focus on describing the representative clinical features of the concept. For explanation, if it exists in the concept name, take into account the detailed items of the concept such as ingredient, dosage form, and strength. If several drugs are contained in a concept, do not explain those drugs separately, but explain the concept name comprehensively and finish the answer with less than 5 sentences.',
    'Measurement': '### Instruction: Briefly explain the clinical background and context of each concept name (measurement) with less than 5 sentences. Do not include sentences that are too ordinary (such as "further details would depend on the specific situation) and focus on describing the representative clinical features of the concept. For explanation, if it exists in the concept name, describe what the decile means clinically.',
    'Procedure': '### Instruction: Briefly explain the clinical background and purpose of each concept name (procedure) with less than 5 sentences. Do not include sentences that are too ordinary (such as "further details would depend on the specific situation) and focus on describing the representative clinical features of the concept.',
}


abspath = str(Path(__file__).resolve().parent.parent.parent)
cpath = os.path.join(abspath, 'data/concepts')
cpath_tmp = os.path.join(abspath, 'data/concepts/tmp')
spath = os.path.join(abspath, 'usedata/descriptions')

categories = ['Condition', 'Drug', 'Measurement', 'Procedure']
co_category = {
    category: pd.read_csv(
        os.path.join(cpath_tmp, category, f'final_concepts_chunk_{args.chunk}.csv')) \
            for category in categories
}

# api 호출시 rate limit에 걸릴 경우 재시도하는 함수 정의
def extract_issues_with_retry(BasePrompt, concept_name, count, retries=30):
    flag = True
    UsePrompt = BasePrompt + '\n' + f"### Concept name: {concept_name} \n"
    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": UsePrompt},
                ]
            )
            return flag, response.choices[0].message.content
        except openai.RateLimitError:
            print(f"Rate limit for chunk #{args.chunk} / {count}-th concept : Attempt {attempt + 1}... Waiting for 20 seconds...")
            time.sleep(20) # 20초간 timesleep
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            sys.exit(1)
        except Exception as e:
            print(f'Error occurrence: {e}')
            traceback.print_exc()
            flag = False
            return flag, None
    
    flag = False
    return flag, "Rate limit exceeded"


for category in categories:
    os.makedirs(os.path.join(spath, category), exist_ok=True)
    description_path = os.path.join(spath, category, f'concept_description_{args.chunk}.csv')
    if os.path.exists(description_path):
        print('Saved file exists.')
        existing_description = pd.read_csv(description_path)
        start_idx = existing_description.shape[0]
        responses = list(existing_description['concept_description'])
    else:
        start_idx = 0
        responses = []

    print(f"Processing {category}_{args.chunk}...")
    count = 1
    for concept_name in tqdm(co_category[category]['concept_name'].iloc[start_idx:]):
        BasePrompt = PromptFor[category]
        flag, response = extract_issues_with_retry(BasePrompt, concept_name, count)
        if response == "Rate limit exceeded":
            sys.exit("Process terminated by rate limit error")
        if flag == False or count % 1000 == 0:
            tmp_descriptions = co_category[category].iloc[:len(responses)]
            tmp_descriptions['concept_description'] = responses
            tmp_descriptions.to_csv(description_path, index=None)
            if flag == False: sys.exit("Error occurred. Save temporary file.")

        responses.append(response)
        count += 1

    co_category[category]['concept_description'] = responses
    co_category[category].to_csv(description_path, index=None)
