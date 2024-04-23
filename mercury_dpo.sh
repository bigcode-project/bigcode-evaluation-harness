# bigcode/starcoder2-3b DPO
accelerate  launch --main_process_port 30011  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-3b-dpo-final_checkpoint  \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-3b-DPO-mercury-result.json

# deepseek-ai/deepseek-coder-1.3b-base DPO
accelerate  launch --main_process_port 30012  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-1.3b-base-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-1.3b-base-DPO-mercury-result.json

# bigcode/starcoder2-7b DPO
accelerate  launch --main_process_port 30013  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-7b-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-7b-DPO-mercury-result.json

# codellama/CodeLlama-7b-hf DPO
accelerate  launch --main_process_port 30014  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-7b-hf-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeLlama-7b-hf-DPO-mercury-result.json

# deepseek-ai/deepseek-coder-6.7b-base DPO
accelerate  launch --main_process_port 30015  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-6.7b-base-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-6.7b-base-DPO-mercury-result.json

# Qwen/CodeQwen1.5-7B DPO
accelerate  launch --main_process_port 30016  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/Qwen/CodeQwen1.5-7B-dpo-final_checkpoint   \
    --load_in_8bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeQwen1.5-7B-DPO-mercury-result.json
 
# bigcode/starcoder2-15b DPO
accelerate  launch --main_process_port 30017  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-15b-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-15b-DPO-mercury-result.json

# codellama/CodeLlama-13b-hf DPO
accelerate  launch --main_process_port 30018  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-13b-hf-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-13b-hf-DPO-mercury-result.json

# deepseek-ai/deepseek-coder-33b-base DPO
accelerate  launch --main_process_port 30019  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-33b-base-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-33b-base-DPO-mercury-result.json

# codellama/CodeLlama-34b-hf DPO
accelerate  launch --main_process_port 30020  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-34b-hf-dpo-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-34b-hf-DPO-mercury-result.json