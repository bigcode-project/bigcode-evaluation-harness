"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""


from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": create_task(True), "humaneval-unstripped": create_task(False)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout

        # Used for Nuggets experiments
        self.one_shot = False
        self.prompt_quality = None
        self.add_context = False
        self.args = None
        self.doc = None

    def get_dataset(self, args):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        self.one_shot = args.one_shot
        self.prompt_quality = args.prompt_quality
        self.add_context = args.add_context
        self.args = args
        return self.dataset["test"]

    def get_start_context(self):
        return "Implement answers to the following questions:\nQUESTION:\n"

    def get_one_shot_example(self):

        # For now, hard-coding the one shot example to be the last task in the HumanEval dataset
        # Also hard-coding the three sample solutions (good, decent, and bad) here
        final_sample = self.dataset["test"][-1]
        correct_sol = final_sample["canonical_solution"]
        really_bad_sol = "    cat: cat cat\n    dog dog dog;\n    return [giraffe if giraffe for giraffe in giraffe]"
        decent_sol = "    lower = 2\n    upper = 8\n    return [i if i % 2 = 0 for i in range(lower, upper)]"
        
        # Create a list of all the different task answers possible for various experiments
        example_task_answers = [really_bad_sol, decent_sol, correct_sol]

        # If add_context is set, add additional context to the prompt. 
        if self.add_context:
            return final_sample["prompt"] + "\nANSWER:\n" + example_task_answers[self.prompt_quality] + "\n" + "QUESTION:\n"
        else:
            return final_sample["prompt"] + "\n" + example_task_answers[self.prompt_quality] + "\n"

    def get_base_prompt(self, doc):
        # Strip prompt if required
        if self.strip_prompt:
            return doc["prompt"].strip()
        else:
            return doc["prompt"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        self.doc = doc
        prompt = self.get_base_prompt(doc)

        # Cases: one shot with context, one shot without context, zero shot
        start_context = self.get_start_context()
        if self.one_shot and self.add_context:
            return start_context + self.get_one_shot_example() + prompt + "\nANSWER:\n"
        elif self.one_shot:
            return self.get_one_shot_example() + prompt
        else:
            return prompt


    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        # Want to remove one shot example and any context when post processing
        prompt = self.get_prompt(self.dataset["test"][idx])
        
        generation = generation[len(prompt) :]

        base_prompt = self.get_base_prompt(self.doc)

        return base_prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        return results
