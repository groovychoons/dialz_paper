import os
import json
import pandas as pd

folder_path = "./data/biaslens"

# Initialize an empty DataFrame to combine all CSVs
combined_df = pd.DataFrame()

# Loop through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path)  # Read each CSV
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)  # Combine

# Select only unique values in the 'Question' field and capitalize the first letter
if 'Question' in combined_df.columns:
    combined_df['Question'] = combined_df['Question'].str.strip().str.capitalize()
    combined_df = combined_df.drop_duplicates(subset='Question')


def add_target_group(df):
    """
    Adds a 'Target Group' column to the DataFrame based on keyword matches in the 'Question' column.
    """
    df['Target Group'] = None

    # Define keywords and their corresponding target group
    target_groups = {
        "gender": ["men", "woman", "women", "female", "females", "girls", "girl", "mother"],
        "race": [
            "races", "racial", "race", "minority", "ethnic", "people of color",
            "african-american", "african american", "black", "darker skinned",
            "hispanic", "latino", "latina", "asian", "indian", "native american"
        ],
        "age": ["old", "older", "young", "younger", "elderly", "senior", "teen", "teenager", "child", "children"],
        "disability": ["disabled", "disabilities", "disability", "handicap", "handicapped", "wheelchair", "blind", "deaf"],
        "sexuality": ["lesbian", "gay", "bisexual", "transgender", "queer", "lgbt", "lgbtq"],
        "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", "sikh", "religion", "religious"],
        "socioeconomic": ["poor", "poverty", "rich", "wealthy", "low-income", "high-income", "middle class"],
    }

    for target, keywords in target_groups.items():
        pattern = "|".join(keywords)  # Create a regex pattern
        df.loc[df['Question'].str.contains(pattern, case=False, na=False), 'Target Group'] = target

    return df

# Apply the function
combined_df = add_target_group(combined_df)

# Print some summary counts
no_target_count = combined_df['Target Group'].isna().sum()
with_target_count = combined_df['Target Group'].notna().sum()
print(f"Count of questions without a target group: {no_target_count}")
print(f"Count of questions with a target group: {with_target_count}")
print("Counts for each target group:")
print(combined_df['Target Group'].value_counts(dropna=False))

# Replace NaNs in 'Target Group' with "none"
combined_df['Target Group'] = combined_df['Target Group'].fillna("none")

# Save the combined DataFrame with target groups for reference
combined_df.to_csv("./data/biaslens/new/biaslens_with_targets.csv", index=False)
print("Saved the DataFrame with target groups to 'biaslens_with_targets.csv'")

# ------------------------------------------------------------------
# **NEW CODE**: Sample up to 500 questions per target group and export to JSON
# ------------------------------------------------------------------

# Identify each unique target group
unique_groups = combined_df['Target Group'].unique()

# Base random_state for reproducibility
base_random_state = 42

# Directory for saving JSON files
output_dir = "./data/biaslens/new"
os.makedirs(output_dir, exist_ok=True)

for idx, group in enumerate(unique_groups):
    # Filter rows for this specific group
    group_df = combined_df[combined_df['Target Group'] == group]
    
    # Determine sample size (up to 500)
    sample_size = min(len(group_df), 500)
    if sample_size == 0:
        print(f"No rows found for target group '{group}'. Skipping.")
        continue
    
    # Sample the data
    group_sample = group_df.sample(n=sample_size, 
                                   random_state=base_random_state + idx, 
                                   replace=False)
    
    # Convert the "Question" column to a Python list
    questions_list = group_sample['Question'].tolist()
    
    # Make a file-friendly group name (replace problematic characters)
    file_friendly_group = str(group).replace(":", "_").replace("/", "_")
    
    # Build output path for JSON
    output_path = os.path.join(output_dir, f"{file_friendly_group}.json")
    
    # Write out the questions as a JSON list
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {sample_size} questions for target group '{group}' to '{output_path}'.")

generic_sample_size = min(len(combined_df), 500)
generic_sample = combined_df.sample(
    n=generic_sample_size, 
    random_state=base_random_state + len(unique_groups), 
    replace=False
)

# Convert to a list
generic_questions_list = generic_sample['Question'].tolist()

# Export as JSON in the same Python-list format
generic_output_path = os.path.join(output_dir, "genericqa.json")
with open(generic_output_path, "w", encoding="utf-8") as f:
    json.dump(generic_questions_list, f, indent=2, ensure_ascii=False)

print(f"Saved {generic_sample_size} random questions from the entire dataset to 'genericqa.json'.")