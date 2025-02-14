import pandas as pd
from data_loader import datasets
models = ['mistral', 'llama', 'qwen']
results = {}

# for model in models:
#     results[model] = {}
#     df = pd.read_csv(f'./results/{model}/mmlu_baseline.csv')
#     avg_result = round(df['baseline_correct'].sum() / len(df), 3)
    
#     results[model] = avg_result

# # Print the results in a table format
# for model in models:
#     print(f"MMLU Baseline({model}): {results[model]}")
#     print()


print("BBQ Baselines:")

category_results = {}

for model in ['mistral', 'llama', 'qwen']:
    category_results[model] = {}
    df = pd.read_csv(f'./results/{model}/bbq_full_baseline.csv')
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['baseline_correct'].sum() / len(category_df), 3) * 100
        category_results[model][category] = avg_result
    total_correct = df['baseline_correct'].sum()
    total_len = len(df)
    category_results[model]['total'] = round(total_correct / total_len, 3) * 100

category_df = pd.DataFrame(category_results)
print(category_df)

print()

print("SVE results:")

category_results = {}

for model in ['qwen']:
    category_results[model] = {}
    df = pd.read_csv(f'./results/{model}/bbq_full_sve.csv')
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['correct'].sum() / len(category_df), 3) * 100
        category_results[model][category] = avg_result
    total_correct = df['correct'].sum()
    total_len = len(df)
    category_results[model]['total'] = round(total_correct / total_len, 3) * 100


category_df = pd.DataFrame(category_results)
print(category_df)

print()

print("Average ISV results:")

results = {}

for model in ['mistral','llama','qwen']:
    results[model] = {}
    df = pd.read_csv(f'./results/{model}/isv_full.csv')
    bbq_avg = round(df['bbq_acc'].sum() / len(df), 3) * 100
    mmlu_avg = round(df['mmlu_acc'].sum() / len(df), 3) * 100
    results[model]['bbq_avg'] = bbq_avg
    results[model]['mmlu_avg'] = mmlu_avg

results_df = pd.DataFrame(results)
print(results_df)


print("Unseen axes results:")

results = {}
for model in ['mistral',]:# 'llama', 'qwen']:
    results[model] = {}
    df = pd.read_csv(f'./results/{model}/bbq_full_baseline.csv')
    results[model]['baseline'] = {}
    for category in ['Race_x_gender', 'Race_x_SES']:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['baseline_correct'].sum() / len(category_df), 3) * 100
        results[model]['baseline'][category] = round(avg_result,3)

    try:
        for _, axis in datasets:
            results[model][axis] = {}
            df = pd.read_csv(f'./results/{model}/isv/{model}_{axis}.csv')
            for category in ['Race_x_gender', 'Race_x_SES']:
                category_df = df[df['category'] == category]
                avg_result = round(category_df['correct'].sum() / len(category_df), 3) * 100
                results[model][axis][category] = round(avg_result, 3)
    except:
        pass
        
    df = pd.read_csv(f'./results/{model}/bbq_full_sve.csv')
    results[model]['sve'] = {}
    for category in ['Race_x_gender', 'Race_x_SES']:
        category_df = df[df['category'] == category]
        avg_result = round(category_df['correct'].sum() / len(category_df), 3) * 100
        results[model]['sve'][category] = round(avg_result,3)

results_df = pd.DataFrame(results)
print(results_df)


print("Discrim-eval results:")
