
import os
import pandas as pd

folder_path = "./data/biaslens_questions"

# Initialize an empty DataFrame to combine all CSVs
combined_df = pd.DataFrame()

# Loop through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path)  # Read each CSV
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)  # Combine into one DataFrame

# Select only unique values in the 'Question' field and capitalize the first letter
if 'Question' in combined_df.columns:
    combined_df['Question'] = combined_df['Question'].str.strip().str.capitalize()
    combined_df = combined_df.drop_duplicates(subset='Question')


def add_target_group(df):
    """
    Adds a 'Target Group' column to the DataFrame based on keyword matches in the 'Question' column.
    """
    # Initialize the 'Target Group' column with NaN
    df['Target Group'] = None

    # Define keywords and their corresponding target group
    target_groups = {
        "gender:women": ["woman", "women", "female", "females", "girls", "girl", "mother"],
        "race:generic": ["races", "racial", "race", "minority", "ethnic", "people of color"],
        "race:black": ["african-american", "african american", "black", "darker skinned"],
        "race:hispanic": ["hispanic", "latino", "latina"],
        "sexuality:generic": ["LGBT", "sexuality", "sexual orientation", "homosexual"],
        "sexuality:gay": ["gay"],
        "intersectional:lesbian": ["lesbian"],
        "intersectional:women_of_color": ["black women", "women of color", "minority women", "ethnic women", "asian women", "hispanic women"]
    }

    # Iterate over each target group and assign it if a keyword is found
    for target, keywords in target_groups.items():
        pattern = "|".join(keywords)  # Create a regex pattern for keywords
        df.loc[df['Question'].str.contains(pattern, case=False, na=False), 'Target Group'] = target

    return df

# Apply the function to your combined DataFrame
combined_df = add_target_group(combined_df)
# Count and print the number of questions without a target group
no_target_count = combined_df['Target Group'].isna().sum()
print(f"Count of questions without a target group: {no_target_count}")

with_target_count = combined_df['Target Group'].notna().sum()
print(f"Count of questions with a target group: {with_target_count}")

print("Counts for each target group:")
print(combined_df['Target Group'].value_counts())

combined_df['Target Group'] = combined_df['Target Group'].fillna("none")
combined_df.to_csv("./data/biaslens_with_targets.csv")