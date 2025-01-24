import sys
import pandas as pd

# Map model names to short names
model_short_names = {
    # "Qwen/Qwen2.5-7B-Instruct": "qwen",
    # "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

for model, short_name in model_short_names.items():
    file_path = f"./data/human_data/{short_name}_human_data.csv"
    df = pd.read_csv(file_path)
    print(f"Loaded data for {model} into DataFrame with shape: {df.shape}")

    # Select only the annotations that are not empty
    df = df.dropna(subset=['baseline_score', 'edited_score'])
    df['score_difference'] = df['edited_score'] - df['baseline_score']
    df['score_average'] = (df['edited_score'] + df['baseline_score']) / 2
    print(f"Processed data for {model} with new columns 'score_difference' and 'score_average'.")

    # Print top 10 rows with highest score differences
    top_10_highest = df.nlargest(10, 'score_difference')
    print(f"Top 10 rows with highest score differences for {model}:\n", top_10_highest)

    # Print top 10 rows with lowest score differences
    top_10_lowest = df.nsmallest(10, 'score_difference')
    print(f"Top 10 rows with lowest score differences for {model}:\n", top_10_lowest)

