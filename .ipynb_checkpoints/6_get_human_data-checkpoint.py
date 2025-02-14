import os
import sys
import datetime
import pandas as pd
from transformers import AutoTokenizer
from dialz import Dataset, ControlModel, ControlVector
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print(datetime.datetime.now())

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Model name must be provided as a command-line argument.")

# Map model names to short names
model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the sample data
sample_df = pd.read_csv("./data/human_data/handwritten_prompts.csv")


def generate_answer(model, vector, coeff, question, type, max_new_tokens=200, repetition_penalty=1.2):

    if type == 'prompting':
        messages = [
                {"role": "system", "content": "Act as if you are extremely woke."},
                {"role": "user", "content": question}
            ]

    else:
        messages = [({"role": "user", "content": question})]


    chat_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",)

    input_ids = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)

    settings = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Greedy decoding
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "return_dict_in_generate": True
    }

    if type == 'baseline' or type == 'prompting':
        model.reset()
        baseline = model.generate(**input_ids, **settings)
        baseline_text = tokenizer.decode(baseline.sequences[0, input_ids["input_ids"].size(1):], skip_special_tokens=True).strip()  
        # Get scores for the baseline
        baseline_score = model.get_activation_score(baseline_text, vector)
        return baseline_text, baseline_score
    
    else:
        model.set_control(vector, coeff)
        edited = model.generate(**input_ids, **settings)
        edited_text = tokenizer.decode(edited.sequences[0, input_ids["input_ids"].size(1):], skip_special_tokens=True).strip()

        edited_score = model.get_activation_score(edited_text, vector)
        # Get scores for the edited text
        return edited_text, edited_score

# Load model based on best BBQ params for race and gender
def get_human_annotation_data():
    dataset = Dataset.create_dataset(
        model_name, 
        ['woke', 'racist'], 
        'race', 
        num_sents=350, 
        system_role="Act as if you are extremely ")
    model = ControlModel(model_name, list(range(-5, -18, -1)))
    vector = ControlVector.train(model, dataset)
    coeff = 1.4
    print(vector.directions.keys())
    for i in range(len(sample_df)):
        if i % 50 == 0:
            print(f"Processing row {i}/{len(sample_df)} at time: {datetime.datetime.now()}")
        
        row = sample_df.iloc[i]
        question = row["prompt"]
        baseline_text, baseline_score = generate_answer(model, vector, coeff, question, 'baseline')
        prompting_text, prompting_score = generate_answer(model, vector, coeff, question, 'prompting')
        edited_text, edited_score = generate_answer(model, vector, coeff, question, 'edited')

        # Update the original DataFrame
        sample_df.loc[row.name, "baseline"] = baseline_text
        sample_df.loc[row.name, "prompting"] = prompting_text
        sample_df.loc[row.name, "edited"] = edited_text
        sample_df.loc[row.name, "baseline_score"] = baseline_score
        sample_df.loc[row.name, "edited_score"] = edited_score
        sample_df.loc[row.name, "prompting_score"] = prompting_score

get_human_annotation_data()

sample_df.to_csv(f"./data/human_data/{model_short_name}_human_data.csv", index=False)