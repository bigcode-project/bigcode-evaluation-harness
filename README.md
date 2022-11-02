<h1 align="center">Code Generation LM Evaluation Harness</h1>


<h4 align="center">
    <p>
        <a href="#features">Tasks</a> |
        <a href="#setup">Usage</a> |
        <a href="#implementing-new-tasks">Contribution</a> |
        <a href="#documentation">Documentation</a> |
        <a href="https://huggingface.co/bigcode">BigCode</a>
    <p>
</h4>

<h3 align="center">
    <img style="float: middle; padding: 10px 10px 10px 10px;" width="50" height="50" src="https://user-images.githubusercontent.com/44069155/191557209-6219acb8-a766-448c-9bd6-284d22b1e398.png" /></a>
</h3>

## Features

This is a framework to evaluate autoregressive code generation language models. This is a work in progress part of the BigCode project, and is insipered from [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating language models in general. We welcome contributions to fix issues, enhance features and add new benchmarks. You can find a contribution guide in [`CONTRIBUTING.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/CONTRIBUTING.md) and more documentation in [`docs/README.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md). 

Below are the features and tasks of this framework:

- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
 - 3 code generation **Python** tasks (with unit tests): [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).
- [CoNaLa](https://huggingface.co/datasets/neulab/conala) for **Python** code generation (2-shot setting and evaluation with BLEU score)
- [Spider](https://huggingface.co/datasets/spider) for **SQL** code generation (2-shot setting and evaluation with BLEU score)
- [Concode](https://huggingface.co/datasets/code_x_glue_tc_text_to_code) for **Java** code generation (2-shot setting and evaluation with BLEU score)
- Code to text task from [CodeXGLUE](https://huggingface.co/datasets/code_x_glue_ct_code_to_text) (zero-shot & fine-tuning) for 6 languages: **Python, Go, Ruby, Java, JavaScript and PHP.** 
- 3 multilingual downstream classification tasks: [Java Complexity prediction](https://huggingface.co/datasets/codeparrot/codecomplex), [Java code equivalence prediction](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench), [C code defect prediction](https://huggingface.co/datasets/code_x_glue_cc_defect_detection).

## Setup

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
```
Install [`torch`](https://pytorch.org/get-started/locally/) based on your device type and the other pacakges using:
```
pip install -r requirements.txt
```
Also make sure you have `git-lfs` installed and are logged in the Hub
```
huggingface-cli login
````

We use [`accelerate`](https://huggingface.co/docs/accelerate/index) to generate code/text in parallel when multiple GPUs are present (multi-GPU mode). You can configure it using:

```bash
accelerate config
```

This evaluation harness can also be used in an an evaluation only mode, you can use a Multi-CPU setting. For this mode you can also find an example of setup instructions in `evaluation_setup.sh`, where we configure the environement and evaluate some MBPP generations donwloaded from the hub.

## Usage
You can use this evaluation harness to generate text solutions to code benchmarks with your model, to evaluate (and execute) the solutions or to do both. While it is betetr to use GPUs for the generation, the evaluation only requires CPUs. So it might be beneficial to separate these two steps. By default both generation and evaluation are performed.

For more details on how to evaluate on the tasks, please refer to the documentation in [`docs/README.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md). 

### Generation and evaluation
Below are some examples to generate and evaluate on some tasks.

```bash
# to evaluate on HumanEval with pass@1, pass@10, pass@100
# (allow_code_execution is for executing the generated code: read the displayed warning)
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution=False 

# to evaluate your model on MBPP with pass@1
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks mbpp \
  --prompt_type_mbpp "incoder" \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution=False 
	
# to evaluate on APPS
# to compute average/strict accuracies: use n_samples 1 
# to compute pass@k: use n_samples != 1 (200)
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks apps \
  --level_apps <LEVEL> \
  --setup_apps <SETUP> \
  --n_samples 1 \
  --temperature 0.1 \
  --batch_size 1 \
  --allow_code_execution=False 

# to evaluate on another task such as code-to-text/conala/spider/concode with BLEU
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks <TASK> \
  --n_samples 1 \
  --temperature 0.1 \
  --batch_size 1 
```

### Generation only

If you want to generate solutions without executing and evaluating the code, set `generation_only` to True, in addition to the instructions above. This will save the solutions in a json file in the working directory.

### Evaluation only

If you already have the generations in a json file from this evaluation harness and want to evaluate them, set `evaluation_only` to True and specify the path of the generations in `generation_path` argument. You might need to reconfigure `accelerate` to use multiple CPUs.  For this mode you can also find an example of setup instructions in `evaluation_setup.sh`. 

Below is an example, be mind of specifying arguments proper to the task you are evaluating on, and note that `model` value here only serves for documenting the experiment.

```
accelerate launch  main.py   --tasks mbpp   --prompt_type_mbpp "incoder" --allow_code_execution=True --evaluation_only True --generations_path generations.json    --model incoder-temperature-08
```

## Implementing new tasks
To implement a new task in this evaluation harness, see the guide in [`docs/guide`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/guide.md). The are also contribution guidelines in this [`CONTRIBUTING.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/CONTRIBUTING.md)

## Documentation
We provide documentation for the existing benchmarks and how we make the evaluation in [`docs/README.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md).

## Remarks
* Currenltly, we use parallel evaluation across multiple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.
* For some scores of ongoing experiments please refer to [`example_scores/README.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/master/example_scores/README.md).

## Acknowledgements
We thank EleutherAI for their work on the [lm-evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) from which this repository is inspired.
