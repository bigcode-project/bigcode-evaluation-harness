# Guide: adding a new task

Here we provide a step by step guide for adding a new task to the `bigcode-evaluation-harness`. For most tasks one needs to define how the prompts should be made in `lm_eval/prompts.py`, define the generation settings, stopping criteria and postprocessing of model outputs in `lm_eval/utils.py`, and the generation settings with how to load and process the reference solutions in `lm_eval/generation.py`.

## Setup

If you haven't already, fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
git checkout -b <task-name>
pip install -e ".[dev]"
```

## Dataset and metric

First add your task to `ALL_TASKS` list in `main.py`. Then go to `lm_eval/evaluator.py`, that's where we load the dataset and run evaluation for each task. in `Evaluator` class, load your dataset and references in `generate_text` method by adding:

```python
elif task == <TASK>:
    dataset = load_dataset(<TASK_DATASET>)
    generations = parallel_generations(
        self.accelerator,
        self.model,
        self.tokenizer,
        dataset,
        mode=<TASK>,
        args=self.args,
        num_tasks=self.args.num_tasks_<TASK>,
    )
    references = get_references_<TASK>(dataset, self.args.num_tasks_<TASK>)
    return generations, references
```
We will see later where to implement `get_references_<TASK>`.

And in `evaluate` method of the class, you can either load your own metric and evaluate it on `generations` and `references` lists or add your task to the BLEU evaluation among other tasks such as conala, concode...

## Prompts and text generation

Now the `parallel_generation` function used in `Evaluator`  needs to work for your task. That includes prompt making, model output postprocessing and generation setting.

* Prompts:
Go to `lm_eval/prompts.py` and implement a function that formats the prompts of your task, it should take as input a datset sample, model prefix and any additional arguments and return the prompt for that sample. In case of few-shot prompts, please save the examples you use in a json file in `lm_eval/few_shot_example`. See `conala_prompt`or `spider_prompt` functions in that file for examples.

* Generation settings:
Go to `lm_eval/utils.py`, and add these two lines for your task to the class `TokenizedDataset` to call the prompt function you implemented (after having imported it in the file):

```python
elif self.mode == <TASK>:
    prompt = <PROMPT_FUNCTION_NAME>(self.dataset[task], prefix="")
```

`complete_code` function generates the text in parallel and returns the processed model predictions, you need to postprocess the model outputs to remove the prompts and only keep the new generated tokens and also clean the output based on your needs. See the other tasks for examples. Also don't forget to add your task to this list in `complete_code` if you use a stopping criteria during the generation

```python
if mode in ["humaneval", "code-to-text", "conala", "spider", "concode"]:
    gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
```

## Refrences and generation settings

in `lm_eval/generation.py` implement a function to get the reference solutions from your dataset and given your number of tasks. If you need different generation settings or stopping criteria, add your task's settings in `parallel_generations`.

## Arguments
Don't forget to add all the new argumets you used to `arguments.py` (for example `num_tasks_<TASK>`)

## Pull request
In your Pull Request, please present the task you want to add and a short description of how the prompts are made and how the generations are evaluated. Also specify if your approach follows the orginal paper's approach or if some changes were introduced. Ideally, you can evaluate some public models and compare the scores to the published results and see if they match.

If there are no published results for your task, make sure the evaluation works properly by testing some samples with a good code generation model such as InCoder-1B. During the experiments you have the option to save `generation.json` and `references.json`, take a look to see if the generations are properely cleaned and are somewhat close to the references for match-based evaluations for example.

