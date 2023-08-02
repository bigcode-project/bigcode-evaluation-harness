
model=NewHope
org=/fsx/loubna/models/
out_path=/fsx/loubna/code/dev/bigcode-evaluation-harness/generations_newhope
task=instruct-humaneval

sbatch -J "eval-$model-$task" /fsx/loubna/code/dev/leader/bigcode-evaluation-harness/leaderboard/instruct_he.slurm "$model" "$task" "$org" "$out_path"


accelerate launch /fsx/loubna/code/bigcode-evaluation-harness/main.py \
    --model $org/$model \
    --tasks instruct-humaneval \
    --max_length_generation 512 \
    --batch_size 50 \
    --n_samples 50 \
    --temperature 0.2 \
    --precision bf16 \
    --allow_code_execution \
    --trust_remote_code \
    --save_generations \
    --use_auth_token \
    --instruction_tokens "### Instruction:\n\n,,\n\n### Response:\n" \
    --save_generations_path $out_path/$model-$task.json

accelerate launch /fsx/loubna/code/bigcode-evaluation-harness/main.py \
    --model WizardLM/WizardCoder-15B-V1.0  \
    --tasks instruct-humaneval \
    --max_length_generation 512 \
    --batch_size 50 \
    --n_samples 50 \
    --temperature 0.2 \
    --precision bf16 \
    --allow_code_execution \
    --trust_remote_code \
    --save_generations \
    --use_auth_token \
    --instruction_tokens "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:,,\n\n### Response:\n" \
    --save_generations_path $out_path/WizardCoder-instruct-humaneval.json

"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:,,\n\n### Response:\n"
echo toekns: $tokens
accelerate launch /fsx/loubna/code/bigcode-evaluation-harness/main.py     --model WizardLM/WizardCoder-15B-V1.0    --tasks instruct-humaneval     --max_length_generation 512     --batch_size 50     --n_samples 50     --temperature 0.2     --precision bf16     --allow_code_execution     --trust_remote_code     --save_generations     --use_auth_token     --instruction_tokens $tokens     --save_generations_path /fsx/loubna/code/bigcode-evaluation-harness/multiple_gens_codegen/$model-$task.json

