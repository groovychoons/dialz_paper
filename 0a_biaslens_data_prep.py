
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
print("Saved the DataFrame with target groups to 'biaslens_with_targets.csv'")

# Define the parent categories
race_categories = ["intersectional:women_of_color", "race:generic", "race:black", "race:hispanic"]
gender_categories = ["intersectional:women_of_color", "gender:women"]


# Sample gender questions first
gender_sample_1 = combined_df[combined_df['Target Group'].isin(gender_categories)].sample(n=500, random_state=42, replace=False)
gender_sample_1['Target Generic'] = 'gender'

# Exclude the sampled gender questions from the remaining DataFrame
remaining_df = combined_df.drop(gender_sample_1.index)

# Sample the second set of gender questions from the remaining DataFrame
gender_sample_2 = remaining_df[remaining_df['Target Group'].isin(gender_categories)].sample(n=500, random_state=43, replace=False)
gender_sample_2['Target Generic'] = 'gender'

# Exclude the second set of sampled gender questions from the remaining DataFrame
remaining_df = remaining_df.drop(gender_sample_2.index)

# Sample race questions from the remaining DataFrame
race_sample_1 = remaining_df[remaining_df['Target Group'].isin(race_categories)].sample(n=500, random_state=44, replace=False)
race_sample_1['Target Generic'] = 'race'

# Exclude the sampled race questions from the remaining DataFrame
remaining_df = remaining_df.drop(race_sample_1.index)

# Sample the second set of race questions from the remaining DataFrame
race_sample_2 = remaining_df[remaining_df['Target Group'].isin(race_categories)].sample(n=500, random_state=45, replace=False)
race_sample_2['Target Generic'] = 'race'


race_sample_1.iloc[:, 2].to_json('./data/race.json', orient='values')
gender_sample_1.iloc[:, 2].to_json('./data/gender.json', orient='values')

# Combine the sampled questions into a single DataFrame
sampled_questions = pd.concat([race_sample_2, gender_sample_2], ignore_index=True)
sampled_questions = sampled_questions.drop(columns=['Role'])

# Save the resulting DataFrame to a CSV file
sampled_questions.to_csv('./data/biaslens_annotation_sample_500.csv', index=False)
print("Sampled questions saved to 'biaslens_annotation_sample_500.csv'.")
