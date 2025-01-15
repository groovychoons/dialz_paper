import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
from tqdm import tqdm

# Define models and configurations
models = {
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Llama Guard": "meta-llama/Llama-Guard-3-8B",
#    "Llama-3-70B": "meta-llama/Llama-3.1-70B-Instruct",
}

# Load the sample data
sample_df = pd.read_csv("./data/biaslens_sample_200.csv")

# Initialize results DataFrame
results_df = sample_df.copy()

# Process each model
for model_name, model_id in models.items():
    print(f"Loading model: {model_name}")
    print(datetime.now())

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    # Generate responses for each prompt
    responses = []
    for prompt in tqdm(sample_df["Question"], desc=f"Generating responses for {model_name}"):
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
        print("Generating response...")
        output = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            num_beams=1,
            do_sample=True
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(decoded_output)

    # Add model outputs to the results DataFrame
    results_df[model_name] = responses

    # Save the results to a new CSV
    results_df.to_csv("./data/model_outputs.csv", index=False)

print("Model outputs saved to 'model_outputs.csv'.")
