# bigcode/starcoder2-3b
accelerate  launch --main_process_port 30001  main.py  \
    --model bigcode/starcoder2-3b   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-3b-mercury-result.json

# deepseek-ai/deepseek-coder-1.3b-base
accelerate  launch --main_process_port 30002  main.py  \
    --model deepseek-ai/deepseek-coder-1.3b-base   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 12   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-1.3b-base-mercury-result.json

# bigcode/starcoder2-7b
accelerate  launch --main_process_port 30003  main.py  \
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

# codellama/CodeLlama-7b-hf 
accelerate  launch --main_process_port 30004  main.py  \
    --model codellama/CodeLlama-7b-hf   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeLlama-7b-hf-mercury-result.json

# deepseek-ai/deepseek-coder-6.7b-base
accelerate  launch --main_process_port 30005  main.py  \
    --model deepseek-ai/deepseek-coder-6.7b-base   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-6.7b-base-mercury-result.json

# Qwen/CodeQwen1.5-7B
accelerate  launch --main_process_port 30006  main.py  \
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

# bigcode/starcoder2-15b
accelerate  launch --main_process_port 30007  main.py  \
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

# codellama/CodeLlama-13b-hf
accelerate  launch --main_process_port 30008  main.py  \
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

# deepseek-ai/deepseek-coder-33b-base
accelerate  launch --main_process_port 30009  main.py  \
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

# codellama/CodeLlama-34b-hf
accelerate  launch --main_process_port 30010  main.py  \
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

# bigcode/starcoder2-3b SFT
accelerate  launch --main_process_port 30011  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-3b-sft-final_checkpoint  \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 10   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-3b-SFT-mercury-result.json

# deepseek-ai/deepseek-coder-1.3b-base SFT
accelerate  launch --main_process_port 30012  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-1.3b-base-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-1.3b-base-SFT-mercury-result.json

# bigcode/starcoder2-7b SFT
accelerate  launch --main_process_port 30013  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-7b-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-7b-SFT-mercury-result.json

# codellama/CodeLlama-7b-hf SFT
accelerate  launch --main_process_port 30014  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-7b-hf-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeLlama-7b-hf-SFT-mercury-result.json

# deepseek-ai/deepseek-coder-6.7b-base SFT
accelerate  launch --main_process_port 30015  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-6.7b-base-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-6.7b-base-SFT-mercury-result.json

# Qwen/CodeQwen1.5-7B SFT
accelerate  launch --main_process_port 30016  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/Qwen/CodeQwen1.5-7B-sft-final_checkpoint   \
    --load_in_8bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path CodeQwen1.5-7B-SFT-mercury-result.json
 
# bigcode/starcoder2-15b SFT
accelerate  launch --main_process_port 30017  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/bigcode/starcoder2-15b-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-15b-SFT-mercury-result.json

# codellama/CodeLlama-13b-hf SFT
accelerate  launch --main_process_port 30018  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-13b-hf-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-13b-hf-SFT-mercury-result.json

# deepseek-ai/deepseek-coder-33b-base SFT
accelerate  launch --main_process_port 30019  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/deepseek-ai/deepseek-coder-33b-base-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path deepseek-coder-33b-base-SFT-mercury-result.json

# codellama/CodeLlama-34b-hf SFT
accelerate  launch --main_process_port 30020  main.py  \
    --model /home/mingzhe/Projects/Mercury/checkpoints/codellama/CodeLlama-34b-hf-sft-final_checkpoint   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 1   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path codeLlama-34b-hf-SFT-mercury-result.json