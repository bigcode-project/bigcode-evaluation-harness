import os
import json
import tempfile

from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer

from arguments import EvalArguments
from lm_eval.evaluator import Evaluator

# TODO add folder for example generations + scores for evaluation mode

TASKS = ["humaneval", "apps", "mbpp", "code-to-text", "conala", "spider", "concode"]
TMPDIR = tempfile.mkdtemp()
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"

def update_args(args):
    args.model = "codeparrot/codeparrot-small"
    args.allow_code_execution = True
    args.save_generations = False
    args.output_path = TMPDIR

    args.batch_size = 1
    args.max_length_generation = 128
    args.do_sample = False
    args.top_p = 0
    args.n_samples = 1
    args.seed = 0

    args.num_tasks_he = 2
    args.num_tasks_apps = 2
    args.num_tasks_mbpp = 2
    args.num_tasks_code_to_text = 2       
    args.num_tasks_concode = 2
    args.num_tasks_conala = 2
    args.num_tasks_spider = 2
    return args


def setup():
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    configPath = os.path.join(TMPDIR, "default_config.yml")
    write_basic_config(save_location=configPath)
    accelerator = Accelerator()
    return model, tokenizer, accelerator


def load_generation_examples(task):
    # generations for testing the generation feature of dummy test model
    with open(f"tests/data/{task}_gen_refs.json") as fp:
        generations = json.load(fp)
    with open(f"tests/data/{task}_gen_gens.json") as fp:
        references = json.load(fp)
    return generations, references

# TEST SUITE
args = update_args(EvalArguments())
model, tokenizer, accelerator = setup()


def test_generation():
    args.generation_only = True
    evaluator = Evaluator(accelerator, model, tokenizer, args)
    for task in TASKS:
        generations, references = evaluator.generate_text(task)
        true_gens, true_refs = load_generation_examples(task)
        assert generations == true_gens
        assert references == true_refs


def test_evaluation():
    # TODO add scores for each task
    args.evaluation_only = True
    for task in TASKS:
        # path to generation examples to evaluate (for which we know the scores)
        args.generations_path(f"tests/data/{task}_eval_gens.json")
        evaluator = Evaluator(accelerator, None, None, args)
        results = evaluator.evaluate(task)
        if task == "mbpp":
            assert results["mbpp"]["mbpp"] == {
                                "pass@1": 0.04344000000000002,
                                "pass@10": 0.21504027174321383,
                                "pass@100": 0.4543711495045223
                                }
        else:
            pass

