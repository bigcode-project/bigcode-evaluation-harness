import json
import os
import warnings

from datasets import load_dataset
from evaluate import load

from lm_eval.generation import (
    parallel_generations,
    get_references_humaneval,
    get_references_mbpp,
    get_references_code_to_text,
)

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.output_path = args.output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

        # evaluation dataset arguments
        self.level_apps = args.level_apps

    def generate_text(self, task):

        if task == "apps":
            dataset = load_dataset(
                "codeparrot/apps", split="test", difficulties=[self.level_apps]
            )
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="apps",
                args=self.args,
                num_tasks=self.args.num_tasks_apps,
            )
            references = None
            return generations, references

        elif task == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
            references = get_references_humaneval(dataset, self.args.num_tasks_he)
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="humaneval",
                args=self.args,
                num_tasks=self.args.num_tasks_he,
            )
            return generations, references

        elif task == "mbpp":
            dataset = load_dataset("mbpp", split="test")
            assert len(dataset) == 500
            # the evaluation set is task ids 11->510
            references = get_references_mbpp(dataset, self.args.num_tasks_mbpp)
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="mbpp",
                args=self.args,
                num_tasks=self.args.num_tasks_mbpp,
            )
            return generations, references

        elif task == "code-to-text":
            dataset = load_dataset(
                "code_x_glue_ct_code_to_text", self.args.language, split="test"
            )
            # we select the first 1200 samples from the test set
            dataset = dataset.select([i for i in range(self.args.code_to_text_data_size)])
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="code-to-text",
                args=self.args,
                num_tasks=self.args.num_tasks_code_to_text,
            )
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            references = get_references_code_to_text(
                dataset, self.args.num_tasks_code_to_text
            )
            return generations, references

        elif task == "conala":
            dataset = load_dataset("neulab/conala", split="test")
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="conala",
                args=self.args,
                num_tasks=self.args.num_tasks_conala,
            )
            n_tasks = (
                self.args.num_tasks_conala
                if self.args.num_tasks_conala is not None
                else len(dataset)
            )
            references = dataset["snippet"][:n_tasks]
            return generations, references

        elif task == "spider":
            dataset = load_dataset("spider", split="validation")
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="spider",
                args=self.args,
                num_tasks=self.args.num_tasks_spider,
            )
            n_tasks = (
                self.args.num_tasks_spider
                if self.args.num_tasks_spider is not None
                else len(dataset)
            )
            references = dataset["query"][:n_tasks]
            return generations, references

        elif task == "concode":
            # concode is the dataset in the text-to-code benchmark of CodeXGLUE
            dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode="concode",
                args=self.args,
                num_tasks=self.args.num_tasks_concode,
            )
            n_tasks = (
                self.args.num_tasks_concode
                if self.args.num_tasks_concode is not None
                else len(dataset)
            )
            references = dataset["code"][:n_tasks]
            return generations, references
        
        elif task == "codexglue-tt":

            dataset = load_dataset(
                    "code_x_glue_tt_text_to_text",self.args.translation_task_codexglue_tt,split="test"
            )
            
            generations = parallel_generations(
                self.accelerator,
                self.model,
                self.tokenizer,
                dataset,
                mode=task,
                args=self.args,
                num_tasks=self.args.num_tasks_codexglue_tt,
            )
            n_tasks = (
                self.args.num_tasks_codexglue_tt
                if self.args.num_tasks_codexglue_tt is not None
                else len(dataset)
            )
            references = dataset["target"][:n_tasks]
            return generations, references

        else:
            raise ValueError(
                f"Task {task} is not supported, please choose from apps, humaneval, mbpp or code-to-text"
            )

    def evaluate(self, task):

        if not self.allow_code_execution and task not in [
            "code-to-text",
            "conala",
            "spider",
            "concode",
            "codexglue-tt"
        ]:
            print(_WARNING)
            raise ValueError(
                "Code evaluation is not enabled. Read the warning above carefully and then use `--allow_code_execution=True` flag to enable code evaluation."
            )
        generations, references = self.generate_text(task)
        if len(generations[0]) != self.args.n_samples:
            generations = [l[:self.args.n_samples] for l in generations]
            warnings.warn("Number of tasks wasn't proportional to number of devices, we removed extra predictions")

        if self.accelerator.is_main_process:
            if not self.args.evaluation_only:
                if self.args.save_generations:
                    with open("generations.json", "w") as fp:
                        json.dump(generations, fp)
                        print("generations were saved")
                if self.args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")
            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if task == "apps":
                code_metric = load("codeparrot/apps_metric")
                results = code_metric.compute(
                    predictions=generations, k_list=[1, 10, 100], level=self.level_apps
                )

            elif task in ["conala", "code-to-text", "spider", "concode","codexglue-tt"]:
                bleu = load("bleu")
                gens = [gen[0] for gen in generations]
                results = bleu.compute(
                    references=references, predictions=gens, max_order=4, smooth=True
                )["bleu"]

            else:
                # HumanEval + MBPP
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                code_metric = load("code_eval")
                results, _ = code_metric.compute(
                    references=references,
                    predictions=generations,
                    num_workers=self.args.num_workers,
                )

            return results
