# Code Generation LM Evaluation Harness

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 code generation Python tasks (with unit tests): [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).
- [CoNaLa](https://huggingface.co/datasets/neulab/conala) for Python code generation (2-shot setting and evaluation with BLEU score)
- [Spider](https://huggingface.co/datasets/spider) for SQL code generation (2-shot setting and evaluation with BLEU score)
- [Concode](https://huggingface.co/datasets/code_x_glue_tc_text_to_code) for Java code generation (2-shot setting and evaluation with BLEU score)
- Code to text task from [CodeXGLUE](https://huggingface.co/datasets/code_x_glue_ct_code_to_text) (zero-shot & fine-tuning) for 6 languages: Python, Go, Ruby, Java, JavaScript and PHP. 
- 3 multilingual downstream classification tasks: [Java Complexity prediction](https://huggingface.co/datasets/codeparrot/codecomplex), [Java code equivalence prediction](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench), [C code defect prediction](https://huggingface.co/datasets/code_x_glue_cc_defect_detection).

## Setup

```bash
git clone https://github.com/bigcode-collaboration/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -r requirements.txt
```
We used `accelerate` to generate code in parallel when multiple GPUs are present. You can configure it using:

```bash
accelerate config
```
## Basic Usage

Below are some examples to evaluate a model (CodeParrot and fine-tuned GPT2 on APPS) on HumanEval and APPS benchmarks:

```bash
# to run both humaneval and apps evaluations on Codeparrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False
	
# to run full pass@k evaluation on the Introductory level of APPS (1000samples)
# the other levels are interview (3000 samples) and competitioon (1000 samples)
accelerate launch main.py \
	--model BigCode/gpt_all_license_apps_finetuned \
	--tasks apps \
	--level_apps introductory \
    	--n_samples 200 \
	--batch_size 40 \
	--temperature 0.6 \
	--allow_code_execution=False

# to evaluate only on X APPS samples using single generations add --num_tasks_apps X --n_samples 1
	
# to evaluate  on some MBPP samples with InCoder 1B (nneds to specify extension)
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

## Remarks
* Currenltly, we use parallel evaluation across multiple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.
* For some scores of ongoing experiments please refer to [example_scores/README.md](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/master/example_scores/README.md).

## Implementing new tasks
To implement a new task in this evaluation harness, see the guide in [docs/guide](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/master/docs/guide.md).

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] test code-to-text for other languages than python
- [ ] test APPS one-shot setup
