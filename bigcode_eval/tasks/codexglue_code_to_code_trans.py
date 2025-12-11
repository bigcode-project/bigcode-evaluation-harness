"""CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Code-to-Code Translation task from CodeXGLUE:
Translating code between Java and C# programming languages.
The dataset is collected from several public repos including Lucene, POI, JGit and Antlr.

Dataset: https://huggingface.co/datasets/google/code_x_glue_cc_code_to_code_trans
- 10,300 training samples
- 500 validation samples  
- 1,000 test samples

This is a zero-shot or few-shot task evaluated with CodeBLEU score.
CodeBLEU is a metric specifically designed for code generation that considers:
- N-gram matching (like BLEU)
- Weighted n-gram matching based on syntax
- Syntax match using AST
- Dataflow match for semantic similarity

Reference: https://arxiv.org/abs/2009.10297
"""
import json

from codebleu import calc_codebleu

from bigcode_eval.base import Task

_CITATION = """
@article{DBLP:journals/corr/abs-2102-04664,
  author    = {Shuai Lu and
               Daya Guo and
               Shuo Ren and
               Junjie Huang and
               Alexey Svyatkovskiy and
               Ambrosio Blanco and
               Colin B. Clement and
               Dawn Drain and
               Daxin Jiang and
               Duyu Tang and
               Ge Li and
               Lidong Zhou and
               Linjun Shou and
               Long Zhou and
               Michele Tufano and
               Ming Gong and
               Ming Zhou and
               Nan Duan and
               Neel Sundaresan and
               Shao Kun Deng and
               Shengyu Fu and
               Shujie Liu},
  title     = {CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding
               and Generation},
  journal   = {CoRR},
  volume    = {abs/2102.04664},
  year      = {2021}
}
"""

# Translation directions supported
# Note: codebleu_lang uses the language identifier expected by the codebleu package
TRANSLATION_DIRECTIONS = {
    "java_cs": {
        "source": "java", 
        "target": "cs", 
        "source_name": "Java", 
        "target_name": "C#",
        "codebleu_lang": "c_sharp",  # Target language for CodeBLEU evaluation
    },
    "cs_java": {
        "source": "cs", 
        "target": "java", 
        "source_name": "C#", 
        "target_name": "Java",
        "codebleu_lang": "java",  # Target language for CodeBLEU evaluation
    },
}


def create_all_tasks():
    """Creates a dictionary of tasks for both translation directions.
    :return: {task_name: task}
        e.g. {codexglue_code_to_code_trans-java_cs: Task, codexglue_code_to_code_trans-cs_java: Task}
    """
    return {
        f"codexglue_code_to_code_trans-{direction}": create_task(direction)
        for direction in TRANSLATION_DIRECTIONS
    }


def create_task(direction):
    class CodeToCodeTransTask(CodeToCodeTrans):
        def __init__(self, **kwargs):
            super().__init__(direction, **kwargs)

    return CodeToCodeTransTask


class CodeToCodeTrans(Task):
    """Code-to-Code Translation task for Java ↔ C# translation.
    
    A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "code_x_glue_cc_code_to_code_trans"
    DATASET_NAME = None

    def __init__(self, direction):
        """Initialize the code translation task.
        
        :param direction: str
            Translation direction, either 'java_cs' or 'cs_java'
        """
        self.direction = direction
        self.direction_config = TRANSLATION_DIRECTIONS[direction]
        super().__init__(
            stop_words=["\n\n", "\n//", "\n/*", "\n#"],  # Stop at blank lines or comments
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle."""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/codexglue_code_to_code_trans_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        return examples[self.direction]

    @staticmethod
    def two_shot_prompt(entry, source_code, examples, source_name, target_name):
        """Two shot prompt format with source and target code examples.
        
        :param entry: str
            Instruction prefix for the task
        :param source_code: str
            The source code to translate
        :param examples: dict
            Few-shot examples containing source1, target1, source2, target2
        :param source_name: str
            Name of the source language (e.g., 'Java')
        :param target_name: str
            Name of the target language (e.g., 'C#')
        :return: str
            The complete prompt
        """
        prompt = f"""{entry}
{source_name}:
{examples['source1']}
{target_name}:
{examples['target1']}

{source_name}:
{examples['source2']}
{target_name}:
{examples['target2']}

{source_name}:
{source_code}
{target_name}:
"""
        return prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        source_name = self.direction_config["source_name"]
        target_name = self.direction_config["target_name"]
        source_field = self.direction_config["source"]
        
        source_code = doc[source_field].strip()
        entry = f"Translate the following code from {source_name} to {target_name}:\n"
        examples = self.fewshot_examples()
        prompt = self.two_shot_prompt(entry, source_code, examples, source_name, target_name)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset).
        
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        target_field = self.direction_config["target"]
        return doc[target_field].strip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        :return: str
        """
        target_name = self.direction_config["target_name"]
        # Extract the generated code after the last target language marker
        marker = f"{target_name}:\n"
        if marker in generation:
            output = generation.split(marker)[-1]
        else:
            output = generation
        
        # Clean up the output - take first complete function/method
        output = output.strip()
        
        # Stop at double newlines or comment markers that might indicate end of function
        for stop in ["\n\n", "\n//", "\n/*"]:
            if stop in output:
                output = output.split(stop)[0]
        
        return output.strip()

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the CodeBLEU metric for the generations.
        
        CodeBLEU combines:
        - ngram_match_score: Standard n-gram matching (like BLEU)
        - weighted_ngram_match_score: N-gram matching weighted by syntax
        - syntax_match_score: AST-based syntax matching
        - dataflow_match_score: Semantic dataflow matching
        - codebleu: Combined score (weighted average of above)
        
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        :return: dict[str: float]
        """
        # Extract the first generation from each list
        predictions = [gen[0] for gen in generations]
        
        # Get the target language for CodeBLEU evaluation
        lang = self.direction_config["codebleu_lang"]
        
        # Compute CodeBLEU score
        # calc_codebleu expects references as list of strings (one per sample)
        # and predictions as list of strings (one per sample)
        results = calc_codebleu(
            references=references,
            predictions=predictions,
            lang=lang,
        )
        
        return results

