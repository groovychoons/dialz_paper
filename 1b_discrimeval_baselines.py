import os
import datetime
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import sys

# Load Hugging Face token
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

print("Loading discrim-eval dataset...")
ds = load_dataset("Anthropic/discrim-eval", "explicit")
df = pd.DataFrame(ds)

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define a function for generating predictions
def generate_baseline(question, max_new_tokens=20, repetition_penalty=1.1):
    input_text = (
        f"Answer with only yes or no.\n\n"
        f"{question}\n"
    )
    messages = [({"role": "user", "content": input_text})]

    chat_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",)
    
    input_ids = tokenizer(chat_input, return_tensors="pt").to(model.device)

    settings = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Greedy decoding
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "return_dict_in_generate": True
    }

    generated_outputs = model.generate(**input_ids, **settings)

    # Extract the newly generated tokens
    new_tokens = generated_outputs.sequences[0, input_ids["input_ids"].size(1):]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Initialize new columns
df["baseline_ans"] = None
df["baseline_prediction"] = None
df["baseline_correct"] = None

for index, row in df.iterrows():
    row = row["train"]
    if index % 500 == 0:
        print(f"Processing row {index}/{len(df)}")
        print("At time:", datetime.datetime.now())

    question = row["filled_template"]

    # Generate the answer
    generated_answer = generate_baseline(question)
    generated_answer = generated_answer.lower()

    # Determine the predicted label
    predicted_label = -1

    for i, letter in enumerate(["no","yes"]):
        if letter in generated_answer[:5]:
            predicted_label = i
            break

    df.at[index, "baseline_ans"] = generated_answer
    df.at[index, "baseline_prediction"] = predicted_label

# Save baseline results
output_dir = f'./results/{model_short_name}'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f'{output_dir}/de_baseline.csv', index=False)

print("Processing complete!")