import os
from datetime import datetime
from pathlib import Path
from subprocess import run
import sys

JOB_COUNT = 0
MODEL_BATCH_SIZE = {
    "bigcode/large-model": 20,
    "huggyllama/llama-7b": 20,
    "huggyllama/llama-13b": 10,
    "huggyllama/llama-30b": 2,
    "Salesforce/codegen-2B-mono": 10,
    "Salesforce/codegen-16B-multi": 8,
    "Salesforce/codegen-16B-mono": 8
}

def get_gen_args(task, model):
    if task in ["humaneval", "humaneval-unstripped"]:
        batch_size = MODEL_BATCH_SIZE.get(model, 50)
        gen_args = f"--max_length_generation 1024 --n_samples 100 --temperature 0.2  --top_p 0.95 --batch_size {batch_size}"
        # batch_size = 1
        # gen_args = f"--max_length_generation 1024 --n_samples 1 --do_sample False --batch_size {batch_size}"
    elif "perturbed-humaneval" in task:
        batch_size = 1
        gen_args = f"--max_length_generation 1024 --n_samples 1 --do_sample False --batch_size {batch_size}"
    else:
        raise ValueError(f"{task} and {model}")
    return gen_args


def main(model_name, model_revision, task):
    global JOB_COUNT
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    num_gpu = 4

    model_id = model_name.split("/")[-1].lower()  # for job-name
    model_revision_arg = f"--revision {model_revision}" if model_revision else ""
    
    gen_args = get_gen_args(task, model_name)

    # launch_command = "python main.py"
    multi_gpu = f"--multi_gpu --num_processes={num_gpu}" if num_gpu > 1 else ""
    launch_command = f"accelerate env && accelerate launch {multi_gpu} main.py"
    output_path = Path(f"/home/toolkit_tmp/evaluation/bigcode/{model_name}/{model_revision}/{task}-100_samples/evaluation_results.json")  # ADJUST

    if "greedy" in str(output_path) or "--do_sample False" in gen_args:
        assert "greedy" in str(output_path) and "--do_sample False" in gen_args

    generations_path = output_path.with_name("generations.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # TF-flags for DS-1000
    job_command = f"""cd /app/bigcode-evaluation-harness && pwd && \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        TF_CPP_MIN_LOG_LEVEL=3 \
        {launch_command} \
        --precision bf16 \
        --model {model_name} {model_revision_arg} \
        --trust_remote_code \
        --use_auth_token \
        --tasks {task} \
        {gen_args} \
        --seed 0 \
        --allow_code_execution \
        --metric_output_path {output_path} \
        --save_generations \
        --save_generations_path {generations_path} \
    """
    toolkit_command = [
        "eai", "job", "submit",
        # "--image", "volatile-registry.console.elementai.com/snow.raymond/bigcode-evaluation-harness:latest-3months",
        # "--image", "volatile-registry.console.elementai.com/snow.raymond/bigcode-evaluation-harness:custom_transformers",
        "--image", "volatile-registry.console.elementai.com/snow.raymond/bigcode-evaluation-harness:raymond_patch-3months",
        "--restartable",
        "--name", f"{task.replace('-', '_')}__{model_id.replace('-', '_').replace('.', '_')}_{JOB_COUNT}__{dt_string}",
        "--data", "snow.raymond.home_tmp:/home/toolkit_tmp",  # ADJUST
        "--data", "snow.code_llm.transformers_cache:/transformers_cache",
        "--env", "HOME=/home/toolkit_tmp",
        "--env", "HF_HOME=/transformers_cache",
        "--cpu", "16",
        "--mem", str(150),
        "--gpu", str(num_gpu),
        "--gpu-mem", "32",
        "--", "bash", "-c", 
        job_command
    ]
    JOB_COUNT += 1

    run(toolkit_command)


if __name__ == "__main__":
    model_name = "bigcode/sc2-1b-ablations"
    
    # Branch-name or commit-id
    # model_revision = ""
    model_revision = "repo_context_Random_8k_8k_vocab-114688_freq_1e6"

    task = "humaneval"

    # for model_revision in []:
    main(model_name, model_revision, task)
