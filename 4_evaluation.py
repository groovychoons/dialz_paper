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

models = ['mistral', 'qwen', 'llama']
conditions = ['race', 'gender', 'racexgender']
results = {}
datasets.append(('all', 'full'))
for model in models:
    results[model] = {}
    for df, name in datasets:
        try:
            df = pd.read_csv(f'./results/{model}/bbq_{name}_baseline.csv')
            avg_result = round(df['baseline_correct'].sum() / len(df), 3) * 100
            
            results[model][name] = avg_result
        except:
            print(f"Error with {model} and {name}")
            continue

df = pd.DataFrame(results)
print(df)
