import pandas as pd
from dialz import Dataset, ControlModel, ControlVector
from transformers import AutoTokenizer

BEST_TRIALS_PATH = "./results/new_best_trials.csv"
best_trials_df = pd.read_csv(BEST_TRIALS_PATH)

MODEL_SHORT_NAMES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}


vectors = []

for _, trial_row in best_trials_df.head(1).iterrows():
    model_short     = trial_row["model"]
    model_name      = MODEL_SHORT_NAMES.get(model_short)

    axis            = trial_row["axis"]
    prompt_type     = trial_row["prompt_type"]
    num_sents       = int(trial_row["num_sents"])
    items_str       = trial_row["items"]
    system_prompt   = trial_row["system_prompt"]
    coeff           = float(trial_row["coeff"])

    items_list = [x.strip() for x in items_str.split(",")]

    print(f"\n=== Loading axis {axis} ===")
    print(f"Model: {model_name}, Axis: {axis}, prompt_type: {prompt_type}, "
            f"num_sents: {num_sents}, items: {items_list}, system_prompt: {system_prompt}, "
            f"coeff: {coeff}")

    # Create dataset
    dataset = Dataset.create_dataset(
        model_name   = model_name,
        items        = items_list,
        prompt_type  = prompt_type,
        num_sents    = num_sents,
        system_role  = system_prompt
    )

    chosen_layer_ids = list(range(-5, -18, -1))

    # Load model
    model = ControlModel(model_name, chosen_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Train vector
    vector = ControlVector.train(model, dataset)

    vectors.append((vector, axis))


for vector, axis in vectors:
    print(f"Vector for axis: {axis}")
    print(vector)