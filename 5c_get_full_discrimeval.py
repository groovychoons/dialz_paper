import datetime
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from dialz import Dataset, ControlModel, ControlVector


# Map model names to short names (if needed)
MODEL_SHORT_NAMES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

# Timestamp for output file naming
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Read the best trials CSV (update path as needed)
BEST_TRIALS_PATH = "./results/best_trials_all.csv"
best_trials_df = pd.read_csv(BEST_TRIALS_PATH)

print("Loading discrim-eval dataset...")
ds = load_dataset("Anthropic/discrim-eval", "explicit")
de = pd.DataFrame(ds)

def generate_answer(model, vector, coeff, tokenizer, question,
                    max_new_tokens=20, repetition_penalty=1.1):
    """
    Given a model and a steering vector, generate a single-letter answer for
    either BBQ or MMLU. 
    """
    input_text = (
        f"Answer with only yes or no.\n\n"
        f"{question}\n"
    )

    messages = [{"role": "user", "content": input_text}]

    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)

    # Set up generation hyperparams
    gen_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Greedy decoding
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "return_dict_in_generate": True,
    }

    # Steering
    model.set_control(vector, coeff)

    generated_outputs = model.generate(**input_ids, **gen_kwargs)
    new_tokens = generated_outputs.sequences[0, input_ids["input_ids"].size(1):]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return answer_text

def get_prediction_with_vector(model, vector, coeff, tokenizer, question):
    """
    Generates the model's answer and converts it to an integer index 
    for either BBQ or MMLU tasks.
    """
    generated_answer = generate_answer(
        model, vector, coeff, tokenizer, question
    )
    generated_answer = generated_answer.lower()

    # Determine the predicted label
    predicted_label = -1

    for i, letter in enumerate(["no","yes"]):
        if letter in generated_answer[:5]:
            predicted_label = i
            break

    return predicted_label, generated_answer


def evaluate_on_discrimeval(model, vector, coeff, tokenizer, df, axis, model_short):
    """
    Evaluate accuracy on a given BBQ subset.
    """
    # Save baseline results
    output_dir = f'./results/{model_short}'
    de_results = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)} at time: {datetime.datetime.now()}")
        
        row = row["train"]
        question = row["filled_template"]
        generated_answer, pred = get_prediction_with_vector(
            model, vector, coeff, tokenizer, question
        )

        de_results.append({
            "decision_question_id": row["decision_question_id"],
            "age": row["age"],
            "gender": row["gender"],
            "race": row["race"],
            "question": question,
            "prediction": pred,
            "answer": generated_answer

        })

    de_results_df = pd.DataFrame(de_results)
    de_results_df.to_csv(f"{output_dir}/{axis}_discrim-eval", index=False)


results = []

# Decide whether or not we are merging
merged_vectors = True # or False
all_axes = True

if merged_vectors:
    for model in best_trials_df["model"].unique():
        print("Merging vectors")
        vector = None
        avg_coeff = 0.0

        for _, trial_row in best_trials_df[best_trials_df["model"] == model].iterrows():

            model_short     = trial_row["model"]
            model_name      = MODEL_SHORT_NAMES.get(model_short)

            axis            = trial_row["axis"]
            prompt_type     = trial_row["prompt_type"]
            num_sents       = int(trial_row["num_sents"])
            items_str       = trial_row["items"]
            system_prompt   = trial_row["system_prompt"]
            coeff           = float(trial_row["coeff"])

            items_list = [x.strip() for x in items_str.split(",")]

            print(f"\n=== Evaluating Best Trial ===")
            print(f"Model: {model_name}, Axis: {axis}, prompt_type: {prompt_type}, "
                f"num_sents: {num_sents}, items: {items_list}, system_prompt: {system_prompt}, "
                f"coeff: {coeff}")

            # Create dataset
            axis_dataset = Dataset.create_dataset(
                model_name   = model_name,
                items        = items_list,
                prompt_type  = prompt_type,
                num_sents    = num_sents,
                system_role  = system_prompt
            )

            # dataset.entries = dataset.entries + axis_dataset.entries
            avg_coeff += coeff

            chosen_layer_ids = list(range(-5, -18, -1))
            model = ControlModel(model_name, chosen_layer_ids)

            # Train vector
            axis_vector = ControlVector.train(model, axis_dataset)
            vector = vector + axis_vector if vector is not None else axis_vector
            del model
            del axis_vector
        
        vector = vector / len(best_trials_df)
        avg_coeff = avg_coeff / len(best_trials_df)

        model = ControlModel(model_name, chosen_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        evaluate_on_discrimeval(model, vector, avg_coeff, tokenizer, de, 'sve', model_short)

        del model

results = []

if all_axes == True:
    # For every single steering vector, calculate the accuracy
    for _, trial_row in best_trials_df.iterrows():
        model_short     = trial_row["model"]
        model_name      = MODEL_SHORT_NAMES.get(model_short)

        axis            = trial_row["axis"]
        prompt_type     = trial_row["prompt_type"]
        num_sents       = int(trial_row["num_sents"])
        items_str       = trial_row["items"]
        system_prompt   = trial_row["system_prompt"]
        coeff           = float(trial_row["coeff"])

        items_list = [x.strip() for x in items_str.split(",")]

        print(f"\n=== Evaluating Best Trial ===")
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

        evaluate_on_discrimeval(model, vector, avg_coeff, tokenizer, de, axis, model_short)
        del model

################################################################################
# 4. Write results to CSV
################################################################################

results_df = pd.DataFrame(results)
output_csv = f"./logs/{timestamp}_best_trials_evaluation.csv"
results_df.to_csv(output_csv, index=False)

print(f"\nDone! Results written to {output_csv}")
