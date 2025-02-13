import numpy as np
import pandas as pd

# Data from the table
bbq_axes = ["Age", "Appearance", "Disability", "Gender", "Nationality", "Race", "Religion", "Sexuality", "Socioeconomic"]

# Baseline, ISV, and SVE values for each model
mistral = {
    "Baseline": np.array([43.9, 52.2, 50.4, 51.6, 55.4, 56.5, 56.5, 49.1, 52.4]),
    "ISV": np.array([55.2, 62.0, 66.4, 63.9, 72.3, 66.2, 66.6, 61.8, 63.7]),
    "SVE": np.array([59.0, 67.3, 65.4, 64.4, 73.6, 71.7, 70.3, 68.3, 69.3]),
}

llama = {
    "Baseline": np.array([62.2, 63.1, 68.4, 66.2, 76.1, 80.7, 75.8, 79.7, 68.9]),
    "ISV": np.array([67.0, 65.1, 74.3, 76.1, 81.8, 84.1, 78.3, 82.5, 74.5]),
    "SVE": np.array([67.9, 66.9, 74.7, 72.6, 82.4, 86.8, 79.9, 81.6, 75.2]),
}

qwen = {
    "Baseline": np.array([74.3, 75.6, 77.6, 77.5, 82.5, 88.6, 78.2, 84.7, 86.0]),
    "ISV": np.array([80.0, 77.1, 79.7, 83.2, 85.3, 91.0, 80.7, 87.4, 89.4]),
    "SVE": np.array([80.6, 77.2, 77.9, 82.1, 83.9, 91.1, 81.1, 86.1, 89.0]),
}

# Function to compute average improvement
def avg_improvement(model_data):
    isv_improvement = np.mean(model_data["ISV"] - model_data["Baseline"])
    sve_improvement = np.mean(model_data["SVE"] - model_data["Baseline"])
    return round(isv_improvement,2), round(sve_improvement,2)

# Compute improvements
mistral_isv, mistral_sve = avg_improvement(mistral)
llama_isv, llama_sve = avg_improvement(llama)
qwen_isv, qwen_sve = avg_improvement(qwen)

# Display results
improvement_results = {
    "Model": ["Mistral", "LLaMA", "Qwen"],
    "Avg ISV Improvement": [mistral_isv, llama_isv, qwen_isv],
    "Avg SVE Improvement": [mistral_sve, llama_sve, qwen_sve],
}

# Convert results into a DataFrame and display
improvement_df = pd.DataFrame(improvement_results)
print(improvement_df)
