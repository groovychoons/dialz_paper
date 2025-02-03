import pandas as pd

# Load datasets
age = pd.read_json('./data/bbq/Age.jsonl', lines=True)
disability = pd.read_json('./data/bbq/Disability_status.jsonl', lines=True)
gender = pd.read_json('./data/bbq/Gender_identity.jsonl', lines=True)
nationality = pd.read_json('./data/bbq/Nationality.jsonl', lines=True)
appearance = pd.read_json('./data/bbq/Physical_appearance.jsonl', lines=True)
race = pd.read_json('./data/bbq/Race_ethnicity.jsonl', lines=True)
religion = pd.read_json('./data/bbq/Religion.jsonl', lines=True)
socioeconomic = pd.read_json('./data/bbq/SES.jsonl', lines=True)
sexuality = pd.read_json('./data/bbq/Sexual_orientation.jsonl', lines=True)

datasets = [
    (age, "age"),
    (appearance, "appearance"),
    (disability, "disability"),
    (gender, "gender"),
    (nationality, "nationality"),
    (race, "race"),
    (religion, "religion"),
    (sexuality, "sexuality"),
    (socioeconomic, "socioeconomic"),
]

racegender = pd.read_json('./data/bbq/Race_x_gender.jsonl', lines=True)
raceSES = pd.read_json('./data/bbq/Race_x_SES.jsonl', lines=True)

# Combine all datasets into one giant dataframe
all_dataframes = [df for df, _ in datasets] + [racegender, raceSES]
bbq_full = pd.concat(all_dataframes, ignore_index=True)