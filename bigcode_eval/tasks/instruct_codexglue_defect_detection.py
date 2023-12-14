"""
CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Defect Detection fron CoedXGLUE
Given a source code, the task is to identify whether it is an insecure code that may attack software systems, 
such as resource leaks, use-after-free vulnerabilities and DoS attack. 
The task is binary classification
"""
from bigcode_eval.base import Task
import re
import evaluate
import json

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10197--10207},
  year={2019}
}
"""


class DefectDetection(Task):
    DATASET_PATH = "code_x_glue_cc_defect_detection"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instruction = '''Is there a defect in the Code, and respond to YES or NO.'''
        code = doc['func']
        prompt = f'''Question: {code}\n{instruction}\n\nAnswer:'''
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return str(int(doc['target']))
    
    @staticmethod
    def contains_keyword(sentence, keywords):
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                return True
        return False

    def postprocess_generation(self, generation, idx):
        # TODO: define the postprocessing for the LM generation
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        true_label = self.get_reference(self.dataset["test"][idx])
        positive_kw=['there is a','ere is a','has a defect','contains a defect']
        negative_kw=['there is no defect']
        process = generation.split("\nAnswer:")[-1]
        short_ans = process.lower().replace(',','.').split('.')[0]
        if 'yes' in short_ans and 'no' not in short_ans:
            prediction = "1" # There is defect
        elif 'no' in short_ans and 'yes' not in short_ans:
            prediction = "0" # There is no defect
        elif self.contains_keyword(process, positive_kw):
            prediction = "1"
        elif self.contains_keyword(process, negative_kw):
            prediction = "0"
        else:
            prediction  = "-1" # invalid
        return {'ID':idx,
                'prediction': prediction,
                'true_label': true_label,
                'raw_text': generation }

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        preds = [gen[0]['prediction'] for gen in generations]
        return  {
            "Accuracy": accuracy_metric.compute(predictions=preds, references=references)['accuracy'],
            "F1(macro)":f1_metric.compute(predictions=preds, references=references,average='macro')['f1']
        }
