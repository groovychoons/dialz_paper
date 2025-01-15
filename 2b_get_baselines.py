import pandas as pd
from dialz import Dataset, ControlModel, ControlVector
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

dataset = Dataset.create_dataset("mistralai/Mistral-7B-Instruct-v0.1", ['sexist', 'feminist'])
model = ControlModel(model_name, list(range(-5, -18, -1)), hf_token)
racism_vector = ControlVector.train(model, dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=model.token)
tokenizer.pad_token_id = 0

sample = pd.read_csv('./data/biaslens_sample_10.csv')


def generate_with_vector_sample(
    input: str,
    vector: ControlVector,
    coeffs: tuple[float, float],
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.1, 
    show_baseline: bool = True,
):
    positive_coeff, negative_coeff = coeffs
    assert positive_coeff > 0
    assert negative_coeff < 0


    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    

    model.reset()
    baseline = tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()


    print("\n++control ---------------------------------------------------")
    model.set_control(vector, positive_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())

    print("\n--control ---------------------------------------------------")
    model.set_control(vector, negative_coeff)
    improved  = tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
    model.reset()

    return baseline, improved

sample["Baseline"] = None
sample["Improved"] = None


for index, row in sample.iterrows():
    question = row["Question"]
    input_text = f"[INST] {question} [/INST]"

    baseline, improved = generate_with_vector_sample(
        input_text,
        racism_vector,
        (1.5, -1.5),
        repetition_penalty=1.3,
    )

    # Save the results back into the DataFrame
    sample.at[index, "Baseline"] = baseline
    sample.at[index, "Improved"] = improved

    
sample.to_csv('./data/biaslens_sample_10_with_baseline.csv', index=False)