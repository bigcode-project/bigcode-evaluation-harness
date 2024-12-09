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
be evaluated using match-based metrics such as BLEU score. However, these metrics fail in capturing the syntactic and 
semantic features of code.  A more appropriate way to evaluate these models is functional correctness, where a solution 
is considered correct if it passes some unit tests, a popular metric for this is `pass@k`. 

In this evaluation harness, we include tasks with unit tests, but also some tasks with BLEU evaluation, due to the scarcity and evaluation cost of the first type.

Before diving into the tasks, here are some instructions that stand for all the benchmarks:
  * Adapt `max_length_generation` based on your model's context size and task, by default it is 512. This value is enough for tasks like HumanEval and MBPP but some tasks such as APPS require a larger value because the prompts are long, you can use the full model's context size.
  * Adapt the  `batch_size` based on your device memory and `n_samples`, by default it is 1. It should be smaller than `n_samples`, but for multiple generations per problem, the larger the batch size the better, since it makes the generation faster.
  * `allow_code_execution` allows the execution of the model-generated (untrusted) code on your machine, please read carefully the displayed warning before calling it (it is off by default). 
  * You can adapt the text generation parameter by changing `do_sample`, `top_p` and `temperature` parameters. 
  * Some models, such as [InCoder](https://huggingface.co/facebook/incoder-6B), might require adding a prefix before the prompt to give a hint about the language. To add the prefix for InCoder to indicate Python language for example, set `prefix` argument to `"<| file ext=.py |>\n"`.
  * The generations are saved with `save_generations` that should be called during the execution, you can visualize the post-processed model generations used for the evaluation. You also have the option of saving the references, it can be useful for tasks that use BLEU score and actual solutions as references, you just need to `save_references`.
  * For experimenting, you can choose the number of tasks to evaluate on instead of using the whole test set with the `limit` argument, try using a number that is proportional to your number of devices.

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
  --allow_code_execution
```

If you want to evaluate only on the first $n$ samples instead of all the test dataset, set `limit` argument to $n$. 

### HumanEval+
[HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus): HumanEval with additional unit tests (80x of the original HumanEval) for each of the 164 problems.

The generation and evaluation follows the same approach as [HumanEval](#humaneval). One only needs to change the task name to `humanevalplus` to run the evaluation on HumanEval+, such as:

```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution
```


### HumanEvalPack

[HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) extends HumanEval to **3** scenarios across **6** languages via human annotations. There are different prompting options depending on the model that can be specified with the `--prompt` flag:
- `continue`: This prompt is the same as HumanEval and only works for HumanEvalSynthesize
- `instruct`: For this prompt an intuitive instruction is given to the model to tell it what to do.
- `octocoder`, `wizardcoder`, `instructcodet5p` etc.: These are custom prompt formats for individual models to align with how they were finetuned.

The three scenarios are listed below. The selectable languages are: `python`, `js`, `java`, `go`, `cpp` & `rust`.
- HumanEvalFix: In this task models are provided with a solution with a subtle bug and several unit tests. The task is to fix the function. There is a variant of this task where the function docstring instead of the unit tests are provided, which can be selected via `humanevalfixdocs`.
```
accelerate launch main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --prompt <PROMPT> \
  --tasks humanevalfixtests-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution
```
- HumanEvalExplain: In this task models need to explain a HumanEval solution (without docstring) and subsequently regenerate the solution given only the model's own explanation. Thus, it requires two runs. The first one generates the descriptions, the second loads the descriptions, generates the solution & is scored.
```
accelerate launch main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --prompt <PROMPT> \
  --tasks humanevalexplaindescribe-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution \
  --generation_only

accelerate launch main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --prompt <PROMPT> \
  --load_data_path <PATH_TO_EXPLANATIONS_FROM_ABOVE> \
  --tasks humanevalexplainsynthesize-python \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution
```
- HumanEvalSynthesize: This is like HumanEval but with human translations for JavaScript, Java, Go, C++ and Rust. It is based on [HumanEval-X](https://arxiv.org/abs/2303.17568), however, with additional fixes and improvements documented [here](https://github.com/bigcode-project/octopack/tree/main/evaluation/create/humaneval-x#modifications-muennighoff).

```
accelerate launch main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --prompt <PROMPT> \
  --tasks humanevalsynthesize-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations
```


There is also a version to run the OpenAI API on HumanEvalPack at `bigcode_eval/tasks/humanevalpack_openai.py`. It requires the `openai` package that can be installed via `pip install openai`. You will need to set the environment variables `OPENAI_ORGANIZATION` and `OPENAI_API_KEY`. Then you may want to modify the global variables defined in the script, such as `LANGUAGE`. Finally, you can run it with `python bigcode_eval/tasks/humanevalpack_openai.py`.


### InstructHumanEval
[InstructHumanEval](https://huggingface.co/datasets/codeparrot/instructhumaneval): 164 handwritten Python programming problems described by an instruction (derived from the HumanEval docstring), a function signature and several unit tests.

This evaluation suite is similar to HumanEval but it is dedicated to instruction-tuned models. Each prompt is built as  an instruction followed by a context, which are separated by delimiter tokens (those used in the instruction-tuning of the model). Here we focus on 3 of such tokens:
- <user_token> : this token represents the role of the person who uses/prompts the model to solve a given task. It can be `Question:`, `USER` etc.
- <end_token> : this token is used to designate the end of the user turn (the end of their request). It can be `<|end|>` or `</s>`. It can even be as simple as `\n`, ` `, or `\n\n`.
- <assistant_token> : similar to <user_token>, this represents the LLM. Some common templates include `Assistant:`, `Response:`, `Answer:`, `<|Assistant|>` etc.

Our evaluation supports two scenarios :
- *Code completion* (`tasks = instruct-humaneval`)
Here the model is prompted with the following instruction
```bash
<user_token> + <instruction> + <end_token> + <assistant_token> + <context>
```
The model is expected to complete a function signature. Make sure to add a `\n` at the end of your `<assistant_token>` to trigger a return to line for the function declaration.
- *Docstring to code* (`tasks = instruct-humaneval-nocontext`)
Here the model is prompted with the following instruction
```bash
<user_token> + <instruction> + <end_token> + <assistant_token>
```
The model is expected to solve the problem formulated as instruction. There is no additional guidance provided by `<context>` (which contains imports, auxiliary functions and the function signature), which increases the complexity of the task.

Here are the commands to run the evaluation in each setting:

for code completion
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks instruct-humaneval \
  --instruction_tokens <user_token>,<end_token>,<assistant_token>\
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution
```

for docstring to code
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks instruct-humaneval-nocontext \
  --instruction_tokens <user_token>,<end_token>,<assistant_token>\
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution
```
The main change is the use of the `instruction_tokens` argument which represents the 3 tokens we mentionned above separated from each other by a comma `,`.
For [StarChat-Beta](https://huggingface.co/HuggingFaceH4/starchat-beta) for example we used these tokens`<|user|>\n,<|end|>\n and <|assistant|>\n`. You might need to escape `|` and `\` characters in bash with `--instruction_tokens \<\|user\|\>$'\n',\<\|end\|\>$'\n',\<\|assistant\|\>$'\n'`
### MBPP
[MBPP](https://huggingface.co/datasets/mbpp):  consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry-level programmers. Each problem consists of a task description in English, a code solution and 3 automated test cases. We evaluate on the test set of samples from index 11 to 511.

* Prompts and generation: We use a few-shot setting in InCoder style prompt: we feed the prompt to the model as a doctring and only include one solution, to help the model catch the function name which is required in the unit tests.
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
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution
```

Low temperatures generally work better for small $k$ in pass@k.

### MBPP+
[MBPP+](https://huggingface.co/datasets/evalplus/mbppplus): MBPP with additional unit tests (35x of the original MBPP) for each of the 164 problems.

The generation and evaluation follows the same approach as [MBPP](#mbpp). One only needs to change the task name to `mbppplus` to run the evaluation on MBPP+, such as:

> [!Note]
> Note MBPP+ only includes **399** tasks which are a subset of the original MBPP dataset (~1000 tasks). 
> The subset is selected from the sanitized MBPP (a subset of ~427 manually examined tasks by the original MBPP authors)
> and EvalPlus further removes low-quality and ill-formed one for benchmark quality control to get MBPP+.

```bash
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks mbppplus \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution
```

By setting `MBPPPLUS_USE_MBPP_TESTS=1` when running MBPP+, one can run the 399 MBPP+ tasks (a subset of the 500 MBPP evaluation tasks) with the original MBPP base tests:

```bash
MBPPPLUS_USE_MBPP_TESTS=1 accelerate launch main.py \
  --tasks mbppplus \
  --allow_code_execution \
  --load_generations_path generations_mbppplus.json \
  --model <MODEL_NAME>
```

### DS-1000
[DS-1000](https://ds1000-code-gen.github.io/): Code generation benchmark with 1000 data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

The task can be specified as `--tasks ds1000-$SUBSET-$MODE`, where subset can include `all` libraries or any of the following subsets: `numpy`, `scipy`, `pandas`, `tensorflow`, `pytorch`, `sklearn`, `matplotlib`. Supported generation modes are `completion` (purely autoregressive) or `insertion` via fill-in-middle [FIM] (this mode now only supports InCoder and BigCode Models).

- Prompts & Generation: prompts include partial code with one or more missing lines. The form of such prompts varies between `completion` and `insertion` modes (`[insert]` token used to reflect FIM region). Default generation args are reflected below.
- Evaluation: generations are evaluated via execution of unit tests. As in the original manuscript, $pass@1$ is evaluated over each of `num_samples` and the mean pass rate is returned as the metric. Default evaluation args are presented below.

Below is the command to run evaluation on the full benchmark in insertion mode with the arguments that correspond to the original manuscript.

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
TF_CPP_MIN_LOG_LEVEL=3 accelerate launch main.py \
  --model <MODEL_NAME> \
  --batch_size <BATCH_SIZE> \
  --tasks ds1000-all-insertion \
  --n_samples 40 \
  --max_length_generation 1024 \
  --temperature 0.2 \
  --top_p 0.95 \
  --allow_code_execution
```

### MultiPL-E
[MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E): is a benchmark for evaluating large language models for code generation that supports 18 programming languages. It takes the OpenAI "HumanEval" Python benchmark and uses little compilers to translate them to other languages. We use similar implementation as [the original repository](https://github.com/nuprl/MultiPL-E/tree/main) and evaluation parameters are similar to HumanEval. Although for this benchmark, we strongly recommend using the provided Dockerfile to build the MultiPL-E container with all required dependencies, and for more safety especially when evaluating on languages like `bash`.
Tasks are named `multiple-<LANG>` where `<LANG>` is the language name, e.g. `multiple-py` for python.

```bash
$ sudo make DOCKERFILE=Dockerfile-multiple all
```
This creates an image called `evaluation-harness-multiple`.

Suppose you generated text with the `bigcode/santacoder` model and saved it in `generations_py.json` with:
```bash
accelerate launch  main.py \
    --model bigcode/santacoder  \
    --tasks multiple-py  \
    --max_length_generation 650 \
    --temperature 0.8   \
    --do_sample True  \
    --n_samples 200  \
    --batch_size 200  \
    --trust_remote_code \
    --generation_only \
    --save_generations \
    --save_generations_path generations_py.json
```
To run the container (here from image `evaluation-harness-multiple`) to evaluate on `generations_py.json`, or another file mount it with `-v`, specify `n_samples` and allow code execution with `--allow_code_execution` (and add the number of problems `--limit`  if it was used during generation):
```bash
$ sudo docker run -v $(pwd)/generations_py.json:/app/generations_py.json:ro -it evaluation-harness-multiple python3 main.py \
    --model bigcode/santacoder \
    --tasks multiple-py \
    --load_generations_path /app/generations_py.json \
    --allow_code_execution  \
    --temperature 0.8 \
    --n_samples 200
```
Execution time may vary depending on the programming languages.

### APPS
[APPS](https://huggingface.co/datasets/codeparrot/apps): is a challenging benchmark for code generation with 10,000 Python problems, 
5,000 for the training and 5000 for the evaluation. It has three difficulty levels: introductory, interview and competition. 
Most papers finetune models on the training split before the evaluation, since the problems are often challenging the problem descriptions are long.
However, Chen et al. evaluated Codex-12B in a one-shot setting, although the details about the prompt format aren't given we propose two evaluation modes: 
with fine-tuning and in a one-shot setting:
* Prompts & generation

**1- Fine-tuning:** we provide the code to fine-tune autoregressive model on this dataset in 
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
Sometimes the prompts can be long and exceed model's context length, so they get truncated. In this case we truncate the prompt before the context length to be able to include the entry "\nUse Call-Based format\nANSWER:\n" for example. The problem description is always at the beginning of the prompt followed by examples that aren't always relevant while the entry is important for the model to know it has to generate Python code that can be executed and not natural text.

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

Below are the commands to run the evaluation with these settings for introductory level for example:
```python
# to compute average/strict accuracies: use n_samples 1 
# to compute pass@k: use n_samples != 1 (200)
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks apps-introductory \
  --n_samples 1 \
  --temperature 0.1 \
  --batch_size 1 \
  --allow_code_execution
```
We expect a model [finetuned](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning/APPS) on the train split of APPS.
TODO: add few-shot setup for APPS.

### Recode
[Recode](https://github.com/amazon-science/recode/tree/main) proposes a set of code and natural language transformations to evaluate the robustness of code-generation models. The perturbations can be applied to any code-generation benchmark. Specifically, they release perturbed versions of HumanEval and MBPP.

For now, we support the perturbed version of the HumanEval benchmark.
The task is specified with `--tasks perturbed-humaneval-{category}-num_seeds_{num_seeds}` where `category` can be one of `format`, `func_name`, `natgen`, `nlaugmenter`, and the number of seeds per perturbation is from `1` to `10`. The author's recommendation is to run with 5 seeds, with greedy generation.

```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation 1024 \
  --tasks <TASK> \
  --batch_size 1 \
  --do_sample False \
  --n_samples 1 \
  --allow_code_execution
```

### StudentEval

[StudentEval](https://huggingface.co/datasets/wellesley-easel/StudentEval) is a 
dataset of 1,749 prompts for 48 problems, authored by 80 students who have only
completed a one-semester Python programming class. Unlike many other benchmarks, 
it has multiple prompts per problem and multiple attempts by the same
participant. Each problem is accompanied by a set of instructor-written test 
cases.

```python
accelerate launch main.py \
  --model <MODEL_NAME> \
  --max_length_generation 512 \
  --tasks studenteval \
  --temperature 0.2 \
  --top_p 0.95 \
  --do_sample True \
  --n_samples 20 \
  --batch_size 20 \
  --allow_code_execution
```

## Mercury
[Mercury](https://huggingface.co/datasets/Elfsong/Mercury) is a Code-LLM computational efficiency benchmark. It comprises 1,889 Python programming tasks with three difficulty stratification, which is divided into two datasets for model evaluation and fine-tuning separately. For each evaluation task, we assign a test case generator to remedy the shortfall of test case coverage. More details can be found in the [paper](https://arxiv.org/abs/2402.07844).

```shell
# Install these libraries before runing Mercury
pip install lctk sortedcontainers
```

```python
accelerate launch main.py  \
    --model <MODEL_NAME>   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path <MODEL_NAME>.json
```

## Code generation benchmarks without unit tests

For these tasks, we do single generations and compare the generated code against reference solutions and compute BLEU score. For the following tasks, we use a two-shot setting where we include 2 inputs and their solutions in the prompt, all preceded by an instruction such as: ` "Answer the following instructions in a one line SQL query:\n"`. The solutions consist of one line so we stop the generation when a new line is generated. 3 languages are present: Python, SQL and Java.

- [CoNaLa](https://huggingface.co/datasets/neulab/conala)for Python code generation, it has 500 tasks in the test set.
- [Spider](https://huggingface.co/datasets/spider) for SQL code generation, it has 1,034 tasks in the test set.
- [Concode](https://huggingface.co/datasets/code_x_glue_tc_text_to_code) for Java code generation, it has 2,000 tasks in the test set.

We only do single generation `n_samples=1`, and use the same generation settings as before.
Below are the commands to run the evaluation:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks <TASK> \
  --n_samples 1 \
  --temperature 0.1 \
  --batch_size 1 
```
If you ever get index out-of-range errors try using a number of problems `limit` that is proportional to the number of devices you are using.

### SantaCoder-FIM
[SantaCoder-FIM](https://huggingface.co/datasets/bigcode/santacoder-fim-task): 4,792 tasks for FIM insertion described in [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988). The tasks are similar to other tasks without unit tests, with two key differences:
1. Instead of BLEU Score, Exact Match is used to score the generations.
2. Use zero-shot setting instead of 2-shot

SantaCoder-FIM includes 2 tasks:
- `StarCoderFIM`: which uses the default FIM tokens `"<fim_prefix>", "<fim_middle>", "<fim_suffix>"`, and
- `SantaCoderFIM`: which uses SantaCoder FIM tokens `"<fim-prefix>", "<fim-middle>", "<fim-suffix>"`
So depending on the FIM tokens used to train the model, you will need to select the appropriate task for evaluation.

We only do single generation `n_samples=1`, and use the same generation settings as before.
Below are the commands to run the evaluation:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks <TASK> \
  --n_samples 1 \
  --temperature 0.2 \
  --batch_size 1 
```
If you ever get index out-of-range errors try using a number of problems `limit` that is proportional to the number of devices you are using.

## Documentation generation task
Code to text task from [CodeXGLUE](https://huggingface.co/datasets/code_x_glue_ct_code_to_text): is a benchmark for English documentation generation from for 6 programming languages: Python, Go, Ruby, Java, JavaScript and PHP. 

For Python: we evaluate in a zero-shot setting. We have two options:
  * in the first one: we give as a prompt the function signature, which we extract by splitting at the beginning of the docstring. This task is `codexglue_code_to_text-python-left`.
  * in the second one: we include the full fucntion body (withoout the docstring) and add this sentence at the end of the prompt: `'\n"""The goal of this function is to:\n'`. This task is `codexglue_code_to_text-python`.
We retrieve the reference solutions from the docstring tokens, similarily to InCoder's approach, since the target docstrings in the dataset include extra context such as argument definitions. We only keep one line in the model generation.

For the other languages (task `codexglue_code_to_text-<language>`): the docstring is not included in the code so we currently don't extract signatures and use the full function body followed by a comment in that language saying `\n=begin The goal of this function is to:\n` for Ruby, and `\n/* The goal of this function is to:\n` for the rest. This task is still not well tested, please report any bugs you might find.

For this task, we advise using greedy generation. For evaluation, we compute the BLEU score.

Below are the commands to run the evaluation:
```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks codexglue_code_to_text-python-left \
  --n_samples 1 \
  --batch_size 1 \
```
## Downstream classification tasks

These are classification tasks for Java and C, we provide the code to finetune models on these benchmarks and evaluate on them in the 
[`finetuning`](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning/) folder:

* [Java Complexity prediction](https://huggingface.co/datasets/codeparrot/codecomplex)
* [Java code equivalence prediction](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench)
* [C code defect prediction](https://huggingface.co/datasets/code_x_glue_cc_defect_detection)

## Natural language reasoning tasks

These are reasoning tasks involving mathematical , symbolic and procedural reasoning with the task description / questions are in natural language.

#### PAL - Program-aided Language Models

In PAL, Large Language Models solve reasoning problems by generating reasoning chains with code. PAL datasets that are currently supported:

* [GSM8K](https://huggingface.co/datasets/gsm8k) - Grade School Math 8K
* [GSM-HARD](https://huggingface.co/datasets/reasoning-machines/gsm-hard) - Created by replacing the numbers in the questions of GSM8K with larger numbers 

The model is prompted with few-shot examples of questions and reasoning steps as code. It then generates reasoning steps for a new question as Python code, which is executed to get the model's predicted answer.

PAL uses two types of few-shot evaluation - 

- `greedy` - samples one generation by greedy decoding and evaluates against reference answers
- `majority_voting` - samples k (k=40 in paper) generations and takes majority voted answer to evaluate against the reference.

**Task signature** : `pal-{dataset_name}-{evaluation_type}` (eg: `pal-gsm8k-greedy`,`pal-gsmhard-majority_voting`)

Commands to run the evaluation:

**Greedy Decoding**

```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks pal-gsm8k-greedy \
  --n_samples 1 \
  --batch_size 1 \
  --do_sample False \
  --allow_code_execution
```

**Majority Voting**

```python
accelerate launch  main.py \
  --model <MODEL_NAME> \
  --max_length_generation <MAX_LENGTH> \
  --tasks pal-gsmhard-majority_voting \
  --n_samples 40 \
  --batch_size 1 \
  --temperature 0.7 \
  --top_p 0.95 \
  --allow_code_execution
```

The complete prompt with 8 shot examples (as used in [PAL](https://github.com/reasoning-machines/pal)) take up `~1500` tokens, hence the value should be greater than that and the recommended value of `max_length_generation` is `2048`.

## How to add a new benchmark

We welcome contributions to add new code benchmarks to this evaluation harness. You can find a step-by-step guide in [`guide.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/guide.md).
