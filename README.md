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

This is a framework to test autoregressive code generation language models. This is a work in progress part of the BigCode project. We welcome contributions to fix issues, enhance features and add new benchmarks. You can find a contribution guide in [CONTRIBUTING.md](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/CONTRIBUTING.md). 

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
git clone https://github.com/bigcode-collaboration/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -r requirements.txt
```
We use `accelerate` to generate code in parallel when multiple GPUs are present. You can configure it using:

```bash
accelerate config
```
## Usage

Below are examples to evaluate code models on some of the available benchmarks:

```bash
# to run both humaneval and apps evaluations on CodeParrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False
	
# to run full pass@k evaluation on the Introductory level of APPS (1000 samples)
# note: to evaluate only on X samples using single generations add --num_tasks_apps X --n_samples 1
accelerate launch main.py \
	--model bigcode-data/gpt_all_license_apps_finetuned \
	--tasks apps \
	--level_apps introductory \
    	--n_samples 200 \
	--batch_size 40 \
	--temperature 0.6 \
	--allow_code_execution=False

	
# to evaluate  on some MBPP samples with InCoder 1B (needs to specify extension)
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks mbpp \
	--num_tasks_mbpp 10 \
	--prompt_type_mbpp "incoder" \
    	--n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False

# to evaluate on another task such as code-to-text/conala/spider/concode
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks code-to-text \
	--num_tasks_code_to_text 20 \
	--n_samples 1 
```

## Implementing new tasks
To implement a new task in this evaluation harness, see the guide in [docs/guide](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/master/docs/guide.md). The are also contribution guidelines in this [CONTRIBUTING.md](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/CONTRIBUTING.md)

## Documentation
We provide documentation for the existing benchmarks and how we make the evaluation in `docs/documentation`.

## Remarks
* Currenltly, we use parallel evaluation across multiple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.
* For some scores of ongoing experiments please refer to [example_scores/README.md](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/master/example_scores/README.md).

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] test code-to-text for other languages than python
- [ ] test APPS one-shot setup
