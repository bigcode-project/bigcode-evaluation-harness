"""
CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Clone Detection (BCB) fron CoedXGLUE
Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others
The task is binary classification
"""
from bigcode_eval.base import Task
import evaluate
import json
import random
from datasets import load_dataset,concatenate_datasets

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}

@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
}
"""


class CloneDetection(Task):
    DATASET_PATH = "code_x_glue_cc_clone_detection_big_clone_bench"
    DATASET_NAME = None

    def __init__(self):
        self.true_label = []
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    @staticmethod
    def sampling(dataset,target,pos_n_sample,neg_n_sample,seed=0):
        random.seed(seed)
        neg_ds = dataset.filter(lambda example: example[target]==False)
        pos_ds = dataset.filter(lambda example: example[target]==True)
        neg_sampled_indices = random.sample(range(0, len(neg_ds)), neg_n_sample)
        pos_sampled_indices = random.sample(range(0, len(pos_ds)), pos_n_sample)
        neg_selected_ds = neg_ds.select(neg_sampled_indices)
        pos_selected_ds = pos_ds.select(pos_sampled_indices)
        combined_ds = concatenate_datasets([neg_selected_ds, pos_selected_ds]).shuffle(seed=seed)
        return combined_ds
    
    @staticmethod
    def contains_keyword(sentence, keywords):
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                return True
        return False

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        select_dataset = self.sampling(self.dataset["test"],"label",1000,1726,seed=0)
        self.true_label = list(map(str, map(int, select_dataset['label'])))
        return select_dataset 

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instruction= '''Is there a clone relation between the Code1 and Code2, and respond to YES or NO.'''
        code1= doc['func1']
        code2= doc['func2']
        prompt= f'''Question: Code1: {code1}.\nCode2: {code2}.\n{instruction}\n\nAnswer:'''
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return str(int(doc['label']))

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
        true_label = self.true_label[idx]
        positive_kw=['there is a','ere is a']
        negative_kw=['there is no']
        process = generation.split("\nAnswer:")[-1]
        short_ans = process.lower().replace(',','.').split('.')[0]
        if 'yes' in short_ans and 'no' not in short_ans:
            prediction = "1" #semantic equivalence (clone)
        elif 'no' in short_ans and 'yes' not in short_ans:
            prediction = "0" #different
        elif self.contains_keyword(process, positive_kw):
            prediction = "1"
        elif self.contains_keyword(process, negative_kw):
            prediction = "0"
        else:
            prediction  = "-1"
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
