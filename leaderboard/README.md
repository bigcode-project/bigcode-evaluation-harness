<h1 align="center">:star: Multilingual Code Evaluation LeaderBoard Guide</h1>


<h4 align="center">
    <p>
        <a href="#running-the-evaluation">Running Evaluation</a> |
        <a href="#submission-of-results-to-the-leaderboard">Results Submission</a> 
    <p>
</h4>

This is a guide to submit and reproduce the numbers in the [Multilingual Code Evaluation LeaderBoard](https://huggingface.co/spaces/bigcode/multilingual-code-evals).
The LeaderBoard is a demo for evaluating and comparing the performance of language models on code generation tasks.

The LeaderBoard is open for submissions of results produced by the community. If you have a model that you want to submit results for, please follow the instructions below.

## Running the evaluation
We report the passs@1 for [HumanEval](https://huggingface.co/datasets/openai_humaneval) Python benchamrk and some languages from the [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) benchmark. We use the same template and parameters for all models.

### 1-Setup
Follow the setup instructions in the evaluation harness [README](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main#setup).

Create two folders `generations_$model` and `metrics_$model` where you will save the generated code and the metrics respectively for your model `$model`.
```bash
cd bigcode-evaluation-harness
mkdir generations_$model
mkdir metrics_$model
```

To run the evaluation, we first generate the code solutions for the target tasks on GPUs, then execute the code on a docker container (only cpus are needed).

### 2- Generation
Below are the instruction for generating the code solutions sequentially or in parallel with slurm. You might need to reduce the batch size for some models or change the precision based on your device.
```bash
# after activating env and setting up accelerate...
langs=(py js java cpp swift php d jl lua r rkt rs)

model=YOUR_MODEL
org=HF_ORGANISATION

for lang in "${langs[@]}"; do
    # use humaneval for py and multipl-e for the rest
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi

    echo "Running task $task"
    generations_path=generations_$model/generations_$task\_$model.json
    accelerate launch main.py \
            --model $org/$model \
            --task $task \
            --n_samples 50 \
            --batch_size 50 \
            --max_length_generation 512 \
            --temperature 0.2 \
            --precision bf16 \
            --trust_remote_code \
            --use_auth_token \
            --generation_only \
            --save_generations_path $generations_path
    echo "Task $task done"
done
```
This will generate and save the code solutions for all tasks in the `generations_$model` folder.

If you want to submit jobs in parallel with `slurm`, run multiple-eval.slurm with:
```bash
langs=(py js java cpp swift php d jl lua r rkt rs)

model=YOUR_MODEL
org=HF_ORGANISATION
out_path=generations_$model

for lang in "${langs[@]}"; do
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi
    echo "Submitting task $task"
    sbatch -J "eval-$model-$task" multiple_evals.slurm "$model" "$task" "$org" "$out_path"
done
```
This will submit one job for each task.

### 3- Execution

We execute and evaluate the solutions inside a docker container, you can either build the image or pull the one we provide:
```bash
# to build it:
# sudo make DOCKERFILE=Dockerfile-multiple all
sudo docker pull ghcr.io/bigcode-project/evaluation-harness-multiple
sudo docker tag ghcr.io/bigcode-project/evaluation-harness-multiple evaluation-harness-multiple
````

Then, you can run the evaluation on the generated code:
```bash
langs=(py js java cpp swift php d jl lua r rkt rs)

model=YOUR_MODEL
org=HF_ORGANISATION
# if you provide absolute paths remove the $(pwd) from the command below
generations_path=generations_$model
metrics_path=metrics_$model

for lang in "${langs[@]}"; do
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi

    gen_suffix=generations_$task\_$model.json
    metric_suffix=metrics_$task\_$model.json
    echo "Evaluation of $model on $task benchmark, data in $generations_path/$gen_suffix"

    sudo docker run -v $(pwd)/$generations_path/$gen_suffix:/app/$gen_suffix:ro  -v $(pwd)/$metrics_path:/app/$metrics_path -it evaluation-harness-multiple python3 main.py \
        --model $org/$model \
        --tasks $task \
        --load_generations_path /app/$gen_suffix \
        --metric_output_path /app/$metrics_path/$metric_suffix \
        --allow_code_execution  \
        --use_auth_token \
        --temperature 0.2 \
        --n_samples 50 | tee -a logs_$model.txt
    echo "Task $task done, metric saved at $metrics_path/$metric_suffix"
done
```

## Submission of results to the LeaderBoard
If you followed the steps above you now have a folder `metrics_$model` with `json` files, each containing the result of one task. To submit the results to the LeaderBoard, you need to create a json summarizing these metrics using `group_jsons.py` and submit it [here](https://huggingface.co/spaces/bigcode/multilingual-code-evals). Follow the instruction on `Submit here` section.
```bash
python group_jsons.py --metrics_path metrics_$model --model $model --org $org --username $your_hf_username
```
For credibility, we invite you to add the generations and json metrics to your submission.

Now you're ready to submit your results by opening a PR on the leaderboard, go to `Submit results :rocket:`section for more details.

## Notes
Some models might require some extra arguments, like [CodeGeeX2-6b](https://huggingface.co/THUDM/codegeex2-6b) which requires providing the language tag as a prefix and doing generation under torch 2.0. And [replit-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) that requires adding extra. You can just add the prefix as a new argument
```bash
# define prefixes base on codegeex-2 repo
declare -A langs
langs=( [py]="# Python" [js]="// JavaScript" [java]="// Java" [cpp]="// C++" [swift]="// Swift" [php]="// PHP" [jl]="# Julia" [lua]="// Lua" [r]="# R" [rkt]="; Racket" [rs]="// Rust" [d]="" )

model="codegeex2-6b"
org="THUDM"

for lang in "${!langs[@]}"; do
    prefix="language: ${langs[$lang]}"
    echo "For language $lang, the prefix is: $prefix"
    generations_path=generations_$model/generations_$task\_$model.json
    accelerate launch main.py \
            --model $org/$model \
            --task multiple-l$ang \
            --n_samples 5 \
            --batch_size 5 \
            --limit 8 \
            --max_length_generation 512 \
            --temperature 0.2 \
            --precision bf16 \
            --trust_remote_code \
            --use_auth_token \
            --generation_only \
            --save_generations_path $generations_path \
            --prefix \"$prefix\" \
    echo "Task $task done"
done
```
Replit model command (pull code from this [PR](https://github.com/bigcode-project/bigcode-evaluation-harness/pull/115)):
```bash
accelerate launch main.py \
    --model replit/replit-code-v1-3b \
    --tasks multiple-$lang \
    --max_length_generation 512 \
    --batch_size 50 \
    --n_samples 10 \
    --temperature 0.2 \
    --precision fp16 \
    --allow_code_execution \
    --trust_remote_code \
    --save_generations \
    --use_auth_token \
    --generation_only \
    --save_generations_path /fsx/loubna/code/bigcode-evaluation-harness/multiple_gens_replit/replit-$lang.json \
    --automodel_kwargs '{\
        \"attn_config\": {\
            \"alibi\": true,\
            \"alibi_bias_max\": 8,\
            \"attn_impl\": \"triton\",\
            \"attn_pdrop\": 0,\
            \"attn_type\": \"multihead_attention\",\
            \"attn_uses_sequence_id\": false,\
            \"clip_qkv\": null,\
            \"prefix_lm\": false,\
            \"qk_ln\": false,\
            \"softmax_scale\": null\
        }\
    }'
```

## Bonus
For the throughput and peak memory measurments, we point you to [optimum-benchamrk](https://github.com/huggingface/optimum-benchmark).
You can follow the instructions in the repo, copy our config yaml and run the command below:
```bash
cp throughput_config.yaml optimum-benchmark/examples
device=cuda:0
batch=1
optimum-benchmark --config-dir examples --config-name throughput_config model=$org/$model device=$device benchmark.input_shapes.batch_size=$batch
```
