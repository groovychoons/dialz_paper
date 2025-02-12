import datetime
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from data_loader import datasets, bbq_full
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
BEST_TRIALS_PATH = "./results/best_trials_qwen.csv"
best_trials_df = pd.read_csv(BEST_TRIALS_PATH)

# Load MMLU
mmlu = load_dataset("cais/mmlu", "all", split="test")
mmlu_df = pd.DataFrame(mmlu)

################################################################################
# 2. Helper Functions
################################################################################

def generate_answer(model, vector, coeff, tokenizer, context, question, answers, task,
                    max_new_tokens=20, repetition_penalty=1.1):
    """
    Given a model and a steering vector, generate a single-letter answer for
    either BBQ or MMLU. 
    """
    if task == "bbq":
        # For 3 answer choices: a, b, c
        input_text = (
            f"Answer with only a single letter.\n\n"
            f"The following are multiple choice questions (with answers):\n"
            f"{context} {question}\n"
            f"(a) {answers[0]}\n"
            f"(b) {answers[1]}\n"
            f"(c) {answers[2]}\n"
        )
    elif task == "mmlu":
        # For 4 answer choices: a, b, c, d
        input_text = (
            f"Answer with only a single letter.\n\n"
            f"The following are multiple choice questions (with answers):\n"
            f"{question}\n"
            f"(a) {answers[0]}\n"
            f"(b) {answers[1]}\n"
            f"(c) {answers[2]}\n"
            f"(d) {answers[3]}\n"
        )
    else:
        raise ValueError("Task must be either 'bbq' or 'mmlu'.")

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

def get_prediction_with_vector(model, vector, coeff, tokenizer,
                               context, question, answers, task):
    """
    Generates the model's answer and converts it to an integer index 
    for either BBQ or MMLU tasks.
    """
    generated_answer = generate_answer(
        model, vector, coeff, tokenizer, context, question, answers, task
    )
    generated_answer = generated_answer.lower()

    if task == "bbq":
        answer_letters = ["a", "b", "c"]
        expanded_letters = ["(a)", "(b)", "(c)"]
    else:  # mmlu
        answer_letters = ["a", "b", "c", "d"]
        expanded_letters = ["(a)", "(b)", "(c)", "(d)"]

    # 1) Try single-letter matching
    predicted_label = -1
    for i, letter in enumerate(answer_letters):
        if letter in generated_answer[:2]:  # e.g. "a" in "a" or "a)" or "a."
            predicted_label = i
            break

    # 2) Try expanded like "(a)", "(b)"
    if predicted_label == -1:
        for i, letter in enumerate(expanded_letters):
            if letter in generated_answer:
                predicted_label = i
                break

    # 3) For BBQ, fallback to searching for full text of answer
    if predicted_label == -1 and task == "bbq":
        for i, ans_text in enumerate(answers):
            if ans_text.lower() in generated_answer:
                predicted_label = i
                break

    return predicted_label

def evaluate_on_bbq(model, vector, coeff, tokenizer, bbq_df):
    """
    Evaluate accuracy on a given BBQ subset.
    """
    bbq_results = []
    total = len(bbq_df)
    correct = 0
    for idx, row in bbq_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{total} at time: {datetime.datetime.now()}")
        context = row["context"]
        question = row["question"]
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        label = row["label"]
        pred = get_prediction_with_vector(
            model, vector, coeff, tokenizer,
            context=context, question=question,
            answers=answers, task="bbq"
        )
        if pred == label:
            correct += 1

        bbq_results.append({
            "example_id": row["example_id"],
            "category": row["category"],
            "context": context,
            "question": question,
            "answers": answers,
            "label": label,
            "prediction": pred,
            "correct": pred == label
        })

    timestamp2 = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    bbq_results_df = pd.DataFrame(bbq_results)
    bbq_results_df.to_csv(f"./logs/{timestamp2}_bbq_results.csv", index=False)
    return correct / total if total > 0 else 0.0

def evaluate_on_mmlu(model, vector, coeff, tokenizer, mmlu_df):
    """
    Evaluate accuracy on MMLU test set.
    """
    total = len(mmlu_df)
    correct = 0
    for idx, row in mmlu_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{total} at time: {datetime.datetime.now()}")
        question = row["question"]
        answers = row["choices"]
        label = row["answer"]  # 0..3
        pred = get_prediction_with_vector(
            model, vector, coeff, tokenizer,
            context="", question=question,
            answers=answers, task="mmlu"
        )
        if pred == label:
            correct += 1
    return correct / total if total > 0 else 0.0

################################################################################
# 3. Main: Iterate over best trials and evaluate
################################################################################
results = []

# Decide whether or not we are merging
merged_vectors = True # or False
merged_datasets = False
all_axes = False

if merged_vectors:
    print("Merging vectors")
    vector = None
    avg_coeff = 0.0
    # dataset = Dataset()

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
    print("Avg coeff:", avg_coeff)
    print(vector)
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        avg_coeff = 2.0

    model = ControlModel(model_name, chosen_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bbq_acc             = evaluate_on_bbq(model, vector, avg_coeff, tokenizer, bbq_full)
    mmlu_acc            = evaluate_on_mmlu(model, vector, avg_coeff, tokenizer, mmlu_df)

    results.append({
        "type":                "merged",
        "model":               model_name,
        "coeff":               avg_coeff,
        "bbq_acc":             round(bbq_acc,3),
        "mmlu_acc":            round(mmlu_acc,3),
    })

    results_df = pd.DataFrame(results)
    output_csv = f"./logs/{timestamp}_merged_vectors_evaluation.csv"
    results_df.to_csv(output_csv, index=False)
    del model

results = []

if merged_datasets:
    avg_coeff = 0.0
    dataset = Dataset()

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
        axis_dataset = Dataset.create_dataset(
            model_name   = model_name,
            items        = items_list,
            prompt_type  = prompt_type,
            num_sents    = num_sents,
            system_role  = system_prompt
        )

        dataset.entries = dataset.entries + axis_dataset.entries
        avg_coeff += coeff
        del axis_dataset

    chosen_layer_ids = list(range(-5, -18, -1))
    model = ControlModel(model_name, chosen_layer_ids)
    vector = ControlVector.train(model, dataset)
    
    avg_coeff = avg_coeff / len(best_trials_df)
    print("Avg coeff:", avg_coeff)
    print("Num entries:", len(dataset.entries))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bbq_acc             = evaluate_on_bbq(model, vector, avg_coeff, tokenizer, bbq_full)
    mmlu_acc            = evaluate_on_mmlu(model, vector, avg_coeff, tokenizer, mmlu_df)

    results.append({
        "type":                "merged",
        "model":               model_name,
        "coeff":               avg_coeff,
        "bbq_acc":             round(bbq_acc,3),
        "mmlu_acc":            round(mmlu_acc,3),
    })
    print(results)

    results_df = pd.DataFrame(results)
    output_csv = f"./logs/{timestamp}_merged_datasets_evaluation.csv"
    results_df.to_csv(output_csv, index=False)

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

        # Evaluate
        axis_df = next((df for df, label in datasets if label == axis), None)

        bbq_axis_acc        = evaluate_on_bbq(model, vector, coeff, tokenizer, axis_df)
        bbq_acc             = evaluate_on_bbq(model, vector, coeff, tokenizer, bbq_full)
        mmlu_acc            = evaluate_on_mmlu(model, vector, coeff, tokenizer, mmlu_df)

        results.append({
            "model":               model_name,
            "axis":                axis,
            "prompt_type":         prompt_type,
            "num_sents":           num_sents,
            "items":               items_str,
            "system_prompt":       system_prompt,
            "coeff":               coeff,
            "bbq_axis_acc":        round(bbq_axis_acc,3),
            "bbq_acc":             round(bbq_acc,3),
            "mmlu_acc":            round(mmlu_acc,3),
        })

################################################################################
# 4. Write results to CSV
################################################################################

results_df = pd.DataFrame(results)
output_csv = f"./logs/{timestamp}_best_trials_evaluation.csv"
results_df.to_csv(output_csv, index=False)

print(f"\nDone! Results written to {output_csv}")
