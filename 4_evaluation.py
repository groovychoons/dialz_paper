import pandas as pd
from data_loader import datasets
models = ['mistral', 'qwen', 'llama']
results = {}

for model in models:
    results[model] = {}
    df = pd.read_csv(f'./results/{model}/mmlu_baseline.csv')
    avg_result = round(df['baseline_correct'].sum() / len(df), 3)
    
    results[model] = avg_result

# Print the results in a table format
for model in models:
    print(f"Results for {model}: {results[model]}")
    print()


print("New results:")

category_results = {}

for model in ['mistral', 'llama']:
    category_results[model] = {}
    df = pd.read_csv(f'./results/{model}/bbq_full_baseline.csv')
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['baseline_correct'].sum() / len(category_df), 3) * 100
        category_results[model][category] = avg_result

category_df = pd.DataFrame(category_results)
print(category_df)


print("SVE results:")

category_results = {}

for model in ['llama']:
    category_results[model] = {}
    df = pd.read_csv(f'./results/{model}/bbq_full_sve.csv')
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['correct'].sum() / len(category_df), 3) * 100
        category_results[model][category] = avg_result

category_df = pd.DataFrame(category_results)
print(category_df)

print("Full amount of correct / len for SVE baseline file:")

for model in ['llama']:
    df = pd.read_csv(f'./results/{model}/bbq_full_baseline.csv')
    total_correct = df['baseline_correct'].sum()
    total_len = len(df)
    print(f"Model: {model}, Total Correct: {total_correct}, Total Length: {total_len}, Ratio: {round(total_correct / total_len, 3) * 100}%")
