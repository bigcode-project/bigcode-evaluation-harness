<h4 align="center">
    <p>
        <a href="#code-generation-benchmarks-with-unit-tests">Benchmarks w/ unit tests</a> |
        <a href="#code-generation-benchmarks-without-unit-tests">Benchmarks w/o unit tests</a> |
        <a href="#documentation-generation-task">Documentation generation </a> |
        <a href="#downstream-classification-tasks">Downstream classification</a> |
        <a href="#how-to-add-a-new-benchmark">New benchmark</a> 
    <p>
</h4>

# Documentation

Here we document the tasks available in this benchmark. Code generation models, just like natural language models can
be evaluated using match-based metrics such as BLEU score. However these metrics fail in capturing the syntactic and 
semantic features of code.  A more appropriate way to evaluate these models is functional correctness, where a solution 
is considered correct if it passes some unit tests, a poplular metric for this is `pass@k`. 

In this evaluation harness we include tasks with unit tests, but also some tasks with BLEU evaluation, due to the scarcity and evaluation cost of the first type.

Before diving into the tasks, here are some instructions that stand for all the benchmarks:
  * Adapt `max_length_generation` based on your model's context size, by default it is 2048.
  * Adapt the  `batch_size` based on your device memory, by default it is 1. The larger the batch size is the better, since it makes the generation faster.
  * `allow_code_execution` allows the execution of the model generated (unstrusted) code on your machine, please read carefully the displayed warning before setting it to `True`. 
  * Some models, such as [InCoder](https://huggingface.co/facebook/incoder-6B), might require adding a prefix before the prompt to give a hint about the language. To add the prefix for InCoder to indicate Python language for example, set `prefix` argument to `"<| file ext=.py |>\n"`.
  * The generations are saved with `save_generations` that is set to True, you can visualize the postprocessed model generations used for the evaluaion. You also have the option of saving the references, it can be useful for tasks that use BLEU score and actual solutions as references, just set `save_references` to True.

## Code generation benchmarks with unit tests

### HumanEval
[HumanEval](https://huggingface.co/datasets/openai_humaneval): 164 handwritten Python programming problems with a function signature, docstring, body, and several unit tests.

* Prompts & generation: in a zero-shot setting, we use function signatures as prompts to the models and generate code until some stop words. By default, top-p sampling is used with $p=0.95$ (same for the other tasks unless we say otherwise), this is set using the arguments `do_sample` and `top_p`. 
We follow Chen et al. approach for pass@k estimation, where $n=200 > k$ solutions are generated per problem for the estimation of the success rate (`n_samples=200`).
* Evaluation: we evaluate the pass@1, pass@10 and pass@100 for a given temperature.

Below are the commands to run the evaluation with these settings:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution=False 
```

If you want to evaluate only on the first $n$ samples instead of all the test dataset, set `num_tasks_he` argument to $n$. 

### MBPP
[MBPP](https://huggingface.co/datasets/mbpp):  consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers. Each problem consists of a task description in English, code solution 
and 3 automated test cases. We evaluate on the test set of samples from index 11 to 511.

* Prompts and generation: We use a few-shot setting and propose two different prompts:
  * Austin et al. (the orginal paper) prompts:
  ````python
  prompt = description + "Your code should satisfy these tests:\n"+ tests"
  ````
  We also give the option to remove the tests since they are the same unit tests used for the evaluation by 
  setting the argument `include_tests_mbpp` to `False`
  * InCoder style prompt: In this case we feed the prompt to the model as a doctring and only include one solution, to help the model catch the 
  function name which is required in the unit tests.
  ```python
  prompt = f'"""\n{description}\n{test_example}\n"""\n'
  ```
  To use this setting (it's the case by default) set `prompt_type_mbpp` to `incoder`. We also give the option to include a code solution in the prompt, just set `include_solution_mbpp` to `True`.
  We use single generations per problem (pass@1), where the model is only given one chance to solve each problem. But we still follow Chen et al. approach similarily to HumanEval for pass@k estimation, we generate $n=15 > k$ solutions ($k=1$ in this case) per problem for the estimation of the success rate (`n_samples=15`).
* Evaluation: we evaluate the pass@1.

Below are the commands to run the evaluation with these settings:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks mbpp \
	--prompt_type_mbpp "incoder" \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution=False \
```

If you want to evaluate only on the first $n$ samples instead of all the test dataset, set `num_tasks_mbpp` argument to $n$. 

Low temperatures generally work better for small $k$ in pass@k.

### APPS
[APPS](https://huggingface.co/datasets/codeparrot/apps): is a challenging benchmark for code generation with 10,000 Python problems, 
5,000 for the training and 5000 for the evaluation. It has three difficulty levels: introductory, interview and competition. 
Most papers finetune models on the training split before the evaluation, since the problems are often challenging the problem descriptions are long.
However, Chen et al. evaluated Codex-12B in a one-shot setting, althought the details about the prompot format aren't given we propose two evaluation modes: 
with fine-tuning and in a one-shot setting:
* Prompts & generation

**1- Fine-tuning:** we provide the code to fine tune autioregressive model on this dataset in 
[`finetuning/APPS`](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning/APPS). To evaluate a fine-tuned model,
we a similar prompt format to the original paper of Hendrycks et al. There are two types of calls based if the function name is provided for the sample or not.

```python
    prompt = "\nQUESTION:\n"
    prompt += sample["question"]
    if starter_code:
        prompt += starter_code
    if fn_name:
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format
    prompt += "\nANSWER:\n"
```
Sometimes the prompts can be long and exceed model's context length, so they get truncated. In this case we truncate the prompt before the context length to be able to include the entry "\nUse Call-Based format\nANSWER:\n" for example. The problem description if always at the beginning of the prompt followed by examples that aren't always relevant while the entry is important for the model to know it has to generate Python code that can be executed and not natural text.

To use this setting (it's the case by default) set the argument `setup_apps` to `finetuning`. To select a difficulty level use `level_apps`argument, by default it is `all`.

**2- Few-shot:** for non finetuned models, we provide one example in the prompt for each call type (Standard Input and Call-Based format). We add the examples with an instruction before the prompt above:

```
    one_shot_prompt = (
        "Implement answers to the following questions:\nQUESTION:\n"
        + examples["problem_type1"]
        + "\nUse Standard Input format\nANSWER:\n"
        + examples["solution_type1"]
        + "\nQUESTION:\n"
        + examples["problem_type2"]
        + "\nUse Call-Based format\nANSWER:\n"
        + examples["solution_type2"]
        + "\n"
        + prompt
    )
```

* Evaluation: we have two types of evaluations for this benchmark:
  * the original Hendrycks et al. evaluation, where we do single generations (`n_samples=1`) and compute the average accuracy of the number 
of tests that pass for each problem, and the sctrict accuracy, where a problem is solved if all tests pass and we average over all problems. This metric is fast to compute given that we do single generations and capture incremental improvement especially for small models. However, strict accuracy is often very low and average accuracy may not very reprsentative as the number of tests is not consistent through the problems. Recent papers evaluate this benchmark using pass@k.
  * we compute the pass@1, pass@10 and pass@100 and generate 200 problems per task (`n_samples=200`). Note that this takes a lot of time since there are 5000 evaluation samples, and there aren't some python stop words for the generation to prevent small models that struggle in answering from generating until max_length or EOS token.

In case of single generations (`n_samples=1`), the first metric is used, but when multiple generations are made the pass@k metric is used.

Below are the commands to run the evaluation with these settings:
```python
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
  --batch_size 10 \
  --allow_code_execution=False 
```
By default <SETUP>="finetuning" and we expect a model [finetuned](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning/APPS) on the train split of APPS, otherwise switch to <SETUP>="fewshot" and a one-shot setting will be used (the scores might be very low since APPS is a challenging benchmark).

If you want to evaluate only on the first $n$ samples instead of all the test dataset, set `num_tasks_apps` argument to $n$. 

## Code generation benchmarks without unit tests

For these tasks, we do single generations and compare the generated code againt reference solutions and compute BLEU score. For the following tasks, we use a two-shot setting where we include 2 inputs and their solutions in the prompt, all preceded by an instruction such as: ` "Answer the following instructions in a one line SQL query:\n"`. The solutions consist of one line so we stop the generation when a new line is generated. 3 languages are present: Python, SQL and Java.

- [CoNaLa](https://huggingface.co/datasets/neulab/conala)for Python code generation.
- [Spider](https://huggingface.co/datasets/spider) for SQL code generation.
- [Concode](https://huggingface.co/datasets/code_x_glue_tc_text_to_code) for Java code generation.

We only do single generation `n_samples=1`, and use the same generation settings as before.
Below are the commands to run the evaluation:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks <TASK> \
  --n_samples 1 \
  --temperature 0.1 \
  --batch_size 10 \
  --allow_code_execution=False 
```
You can evaluate on the first $n$ samples of a `<TASK>` by setting `num_tasks_<TASK>` to $n$.

## Documentation generation task
Code to text task from [CodeXGLUE](https://huggingface.co/datasets/code_x_glue_ct_code_to_text): is a benchmark for english documentation generation from for 6 programming languages: Python, Go, Ruby, Java, JavaScript and PHP. 

For Python: we do the evaluation in a zero-shot setting. We have two options:
  * in the first one: we give as a prompt the function signature, that we extract it by splitting at the beginning of the docstring. 
  * in the second one: we include the full fucntion body (withoout the docstring) and add this sentence at the end of the prompt: `'\n"""Explanation of the code above:\n'`.
We retrieve the reference solutions from the docstring tokens, similarily to InCoder's approach, since the target docstrings in the dataset include extra context such as argument definitions. We only keep one line in the model generation.

For the other languages : the docstring is not included in the code so we currently don't extract signatures and use the full function body followed by a comment in that language saying `\n=begin Explanation of the code above:\n` for Ruby, and `\n/* Explanation of the code above:\n` for the rest. This task is still not well tested, please report any bugs you might find.

For this task we use greedy generation, and we compute the BLEU score.  We evaluate on the first 2,000 examples from the test set. You can select the language by setting `langauge`argument and choose the prompt type as function signature for Python (it's the case by defaullt) by setting the argument `prompt_type` to `left`.

## Downstream classification tasks

These are classification tasks for Java and C, we provide the code to finetune models on these benchmarks and evaluate on them in the 
[`finetuning`](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning/) folder:

* [Java Complexity prediction](https://huggingface.co/datasets/codeparrot/codecomplex)
* [Java code equivalence prediction](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench)
* [C code defect prediction](https://huggingface.co/datasets/code_x_glue_cc_defect_detection)

## How to add a new benchmark

We welcome contribution to add new code benchmarks to this evaluation harness. You can find a step by step guide in [`guide.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/guide.md).

## To do:
- [ ] add execution commands
- [ ] add links and references
