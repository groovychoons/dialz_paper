import pandas as pd

# result = df.head(5000)['baseline_correct'].sum() / 5000
# print(result)

# result_ambig = df[df['context_condition'] == 'ambig'].head(5000)['baseline_correct'].sum() / 2500
# print(result_ambig)

# result_disambig = df[df['context_condition'] == 'disambig'].head(5000)['baseline_correct'].sum() / 2500
# print(result_disambig)

models = ['mistral', 'qwen', 'llama']
results = {}

for model in models:
    results[model] = {}
    df = pd.read_csv(f'./results/{model}/mmlu_baseline.csv')
    avg_result = df['baseline_correct'].sum() / len(df)
    
    results[model] = avg_result

# Print the results in a table format
for model in models:
    print(f"Results for {model}: {results[model]:.4f}")
    print()


print("New results:")

models = ['mistral', 'qwen', 'llama']
conditions = ['gender', 'race', 'racexgender']
results = {}

for model in models:
    results[model] = {}
    for condition in conditions:
        df = pd.read_csv(f'./results/{model}/bbq_{condition}_baseline.csv')
        avg_result = df['baseline_correct'].sum() / len(df)
        
        results[model][condition] = avg_result

df = pd.DataFrame(results)
print(df)
