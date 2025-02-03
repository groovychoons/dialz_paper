
# python 6_get_human_data.py "mistralai/Mistral-7B-Instruct-v0.1"
bash 1_run_bbq_baselines.sh
python 5_get_full_steered.py

# Run the steering for llama and qwen
bash 3_run_steering.sh

