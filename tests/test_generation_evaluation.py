import json
import os
import tempfile

from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator

# TODO add more tasks

# Tasks for generation test
GEN_TASKS = ["humaneval", "mbpp"]
# Tasks for evaluator tests
EVAL_TASKS = ["humaneval", "mbpp", "pal-gsm8k-greedy"]
TMPDIR = tempfile.mkdtemp()
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"
REF_EVAL_SCORES = {
    "humaneval": {"pass@1": 0.25},
    "mbpp": {"pass@1": 0.25},
    "pal-gsm8k-greedy": {"accuracy": 1.0, "num_failed_execution": 0},
}


def update_args(args):
    args.model = "hf-internal-testing/tiny-random-gpt2"
    # the executed code for the tests is safe (see tests/data/*_eval_gens.json)
    args.allow_code_execution = True
    args.save_generations = False
    args.save_generations_path = ""
    args.save_references = False
    args.save_references_path = ""
    args.metric_output_path = TMPDIR
    args.load_generations_path = None
    args.generation_only = False
    args.check_references = False
    # postprocessing for HumanEval and MBPP makes generations
    # with dummy model not distinctive
    args.postprocess = False
    args.instruction_tokens = None

    args.limit = 2
    args.limit_start = 0
    args.batch_size = 1
    args.max_length_generation = 300
    args.left_padding = False
    args.do_sample = False
    args.top_p = 0
    args.n_samples = 1
    args.seed = 0
    args.prompt = None
    args.precision = None
    args.modeltype = None
    args.max_memory_per_gpu = None
    return args


def setup():
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    configPath = os.path.join(TMPDIR, "default_config.yml")
    write_basic_config(save_location=configPath)
    accelerator = Accelerator()
    return model, tokenizer, accelerator


def load_generation_examples(task):
    # generations for testing the generation feature of dummy test model
    with open(f"tests/data/{task}_gen_gens.json") as fp:
        gens = json.load(fp)
    with open(f"tests/data/{task}_gen_refs.json") as fp:
        refs = json.load(fp)
    return gens, refs


args = update_args(EvalArguments())
set_seed(args.seed)
model, tokenizer, accelerator = setup()


def test_generation():
    args.generation_only = True
    args.save_every_k_tasks = -1
    evaluator = Evaluator(accelerator, model, tokenizer, args)
    for task in GEN_TASKS:
        print(f"testing task {task}")
        generations, references = evaluator.generate_text(task)
        true_gens, true_refs = load_generation_examples(task)
        assert generations == true_gens
        assert references == true_refs
    print("passed gen")


def test_evaluation():
    # TODO add scores for each task
    args.n_samples = 2
    args.save_every_k_tasks = -1
    for task in EVAL_TASKS:
        print(f"testing task {task}")
        # path to generation examples to evaluate
        args.load_generations_path = f"tests/data/{task}_eval_gens.json"
        evaluator = Evaluator(accelerator, None, None, args)
        results = evaluator.evaluate(task)
        assert results == REF_EVAL_SCORES[task]
    print("passed eval")
