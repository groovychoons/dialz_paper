import json

def create_dataset_entries(data, bias_type_filter):
    """
    Given the parsed JSON data and a bias_type_filter (e.g. "race" or "gender"),
    returns a list of dictionaries with "positive" and "negative" keys.
    
    - positive = context + stereotype sentence
    - negative = context + anti-stereotype sentence
    """
    dataset_entries = []

    # Navigate into the JSON structure
    intersentence_entries = data["data"].get("intersentence", [])

    for entry in intersentence_entries:
        if entry.get("bias_type") == bias_type_filter:
            context = entry.get("context", "").strip()
            sentences = entry.get("sentences", [])

            # Find stereotype and anti-stereotype sentences
            stereotype_sents = [
                s for s in sentences if s.get("gold_label") == "stereotype"
            ]
            anti_stereotype_sents = [
                s for s in sentences if s.get("gold_label") == "anti-stereotype"
            ]

            # In many cases you might have multiple. For simplicity, pick the first of each type.
            if stereotype_sents and anti_stereotype_sents:
                stereotype_sentence = stereotype_sents[0].get("sentence", "").strip()
                anti_stereotype_sentence = anti_stereotype_sents[0].get("sentence", "").strip()

                positive_str = f"{context} {stereotype_sentence}"
                negative_str = f"{context} {anti_stereotype_sentence}"

                dataset_entries.append({
                    "positive": positive_str,
                    "negative": negative_str
                })

    dataset_entries = dataset_entries[:500]
    print(len(dataset_entries))
    return dataset_entries


input_filename = "./data/stereoset/dev.json"
with open(input_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Create separate datasets
race_entries = create_dataset_entries(data, bias_type_filter="race")
gender_entries = create_dataset_entries(data, bias_type_filter="gender")

# 3. Write out the results to two JSON files
#    The output format is a list of { "positive": "...", "negative": "..." } dictionaries.
with open("./data/stereoset/stereoset_race.json", "w", encoding="utf-8") as f:
    json.dump(race_entries, f, indent=2, ensure_ascii=False)

with open("./data/stereoset/stereoset_gender.json", "w", encoding="utf-8") as f:
    json.dump(gender_entries, f, indent=2, ensure_ascii=False)

print("Done! Created stereoset_race.json and stereoset_gender.json.")
