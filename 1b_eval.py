import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from dialz import Dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Read in all 3 files
gender = pd.read_json('./data/bbq/BBQ_Gender_identity.jsonl', lines=True)
race = pd.read_json('./data/bbq/BBQ_Race_ethnicity.jsonl', lines=True)
racexgender = pd.read_json('./data/bbq/BBQ_Race_x_gender.jsonl', lines=True)

data_files = [("gender", gender), ("race", race), ("racexgender", racexgender)]

# Load LLM
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)

## Get baseline first
def generate_baseline(context, question, answers, max_new_tokens=50, repetition_penalty=1.1):
    input_text = f"[INST]{context} {question}\n(a) {answers[0]}\n(b) {answers[1]}(c) {answers[1]}[/INST]"
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Greedy decoding
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    generated_ids = model.generate(**input_ids, **settings)
    return tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True).strip()



unknowns = ["not specified", "not clear", "not possible", "not appropriate", "cannot be determined"]

## for all 3 files
for name, df in data_files:
    # Evaluate predictions
    df.drop(columns=["additional_metadata"], inplace=True)

    # df["baseline_ans"] = None
    # df["baseline_prediction"] = None
    # df["baseline_correct"] = None

    # results = []
    # for index, row in df.head(3).iterrows():
    #     print(name, index)
    #     context = row["context"]
    #     question = row["question"]
        
    #     # Extract answers from the "answer_info" column
    #     answer_info = row["answer_info"]
    #     answers = [answer_info["ans0"][0], answer_info["ans1"][0], answer_info["ans2"][0]]

    #     # Generate the answer
    #     generated_answer = generate_baseline(context, question, answers)
    #     generated_answer = generated_answer.split('[/INST]')[-1].strip()

    #     # Determine the predicted label
    #     predicted_label = -1

    #     # Check for unknown matches if no direct match is found
    #     for i in range(len(answers)):
    #         # Check if this answer's second element is "unknown"
    #         if answer_info[f"ans{i}"][1].lower() == "unknown":
    #             # If the generated answer matches any of the unknowns, assign this index
    #             if any(unk in generated_answer.lower() for unk in unknowns):
    #                 predicted_label = i
    #                 break
        
    #     # Check for direct matches
    #     if predicted_label == -1:
    #         for i, answer in enumerate(answers):
    #             if answer in generated_answer:
    #                 predicted_label = i
    #                 break

    #     correct_label = row["label"]

    #     df.at[index, "baseline_ans"] = generated_answer
    #     df.at[index, "baseline_prediction"] = predicted_label
    #     df.at[index, "baseline_correct"] = (predicted_label == correct_label)

    # # Save baseline results
    # df.to_csv(f'./data/bbq/{name}_baseline.csv', index=False)

    # # Drop the columns
    # df.drop(columns=["baseline_ans", "baseline_prediction", "baseline_correct"], inplace=True)



## Load different datasets via dialz
generic_250 = Dataset.create_dataset(model_name, ["racist", "anti-racist"], "generic", 2)
related_250 = Dataset.create_dataset(model_name, ["racist", "anti-racist"], "starters", 2)
starters_250 = Dataset.create_dataset(model_name, ["racist", "anti-racist"], "race", 2)
stereoset_250 = Dataset.load_corpus(model_name, "stereoset_race", 2)

datasets = [generic_250, related_250, starters_250, stereoset_250]
# Add another for loop 
for dataset in datasets:
    print("Dataset:")
    print(dataset)

# Create model via dialz


# Output results for each steering vector (4), each dataset (3)

# Save them