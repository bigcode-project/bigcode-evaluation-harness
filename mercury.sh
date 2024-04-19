accelerate  launch --main_process_port 30007  main.py  \
    --model deepseek-ai/deepseek-coder-1.3b-base   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-1.3b-base-mercury-result.json

accelerate  launch --main_process_port 30002  main.py  \
    --model bigcode/starcoder2-7b   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-7b-mercury-result.json

accelerate  launch --main_process_port 30002  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-3b-sft-final_checkpoint  \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-3b-sft-mercury-result.json

accelerate  launch --main_process_port 30001  main.py  \
    --model codellama/CodeLlama-7b-hf   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeLlama-7b-hf-mercury-result.json


accelerate  launch --main_process_port 30002  main.py  \
    --model deepseek-ai/deepseek-coder-6.7b-base   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-6.7b-base-mercury-result.json


accelerate  launch --main_process_port 30003  main.py  \
    --model Qwen/CodeQwen1.5-7B   \
    --load_in_8bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeQwen1.5-7B-mercury-result.json \
    --save_generations_path CodeQwen1.5-7B-mercury-generations.json


accelerate  launch --main_process_port 30004  main.py  \
    --model bigcode/starcoder2-15b   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-15b-mercury-result.json

accelerate  launch --main_process_port 30005  main.py  \
    --model codellama/CodeLlama-13b-hf   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-13b-hf-mercury-result.json

accelerate  launch --main_process_port 30006  main.py  \
    --model deepseek-ai/deepseek-coder-33b-base   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-33b-base-mercury-result.json

accelerate  launch --main_process_port 30007  main.py  \
    --model codellama/CodeLlama-34b-hf   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-34b-hf-mercury-result.json







