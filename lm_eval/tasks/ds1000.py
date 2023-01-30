"""
DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

https://arxiv.org/pdf/2211.11501.pdf

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

Homepage: https://ds1000-code-gen.github.io/
"""

import io, itertools, functools, pathlib, requests, zipfile
from lm_eval.base import Task

_CITATION = """
@article{Lai2022DS1000,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Yuhang Lai and Chengxi Li and Yiming Wang and Tianyi Zhang and Ruiqi Zhong and Luke Zettlemoyer and Scott Wen-tau Yih and Daniel Fried and Sida Wang and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.11501}
}
"""


def get_tasks():
    def _get_task(key):
        class DS1000(DS1000General):
            def __init__(self):
                super().__init__(key)

        return DS1000

    return {
        f"ds1000-{key.lower()}": _get_task(key)
        for key in [
            "All",
            "Numpy",
            "Pandas",
            "Scipy",
            "Matplotlib",
            "Sklearn",
            "Tensorflow",
            "Pytorch",
        ]
    }


class DS1000General(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, key):
        super().__init__(
            stop_words=["</code>", "END SOLUTION", "\n\n"],
            requires_execution=True,
        )
        self._key = key
        self._dir = pathlib.Path(__file__).parent / "ds"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._src = self._dir / "ds1000.py"
        self._data = self._dir / "ds1000_data"
        self._download_source()
        self._download_dataset()

    def _download_source(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/main/ds1000.py?raw=true"
        if not self._src.exists():
            print("Downloading source code...")
            r = requests.get(url, stream=True)
            with open(self._src, "wb") as f:
                f.write(r.content)
            with open(self._src.parent / "__init__.py", "w") as f:
                f.write("")
            print("Done.")

    def _download_dataset(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/main/ds1000_data.zip?raw=true"
        if not (self._data).exists():
            print("Downloading dataset...")
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self._dir)
            print("Done.")

    @functools.lru_cache()
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        from .ds.ds1000 import DS1000Dataset

        data = DS1000Dataset(self._data, mode="Completion").data
        if self._key == "All":
            dataset = list(itertools.chain(*data.values()))
        else:
            dataset = data[self._key]
        return dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["prompt"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["reference_code"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        processed = generation.split("<code>")[-1]
        for stop in self.stop_words:
            try:
                processed = processed.split(stop)[0]
            except IndexError:
                continue
        return processed.strip()

    def process_results(self, generations, references):
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
        dataset = self.get_dataset()
        num_correct = 0
        for i, ref in enumerate(references):
            test = [doc for doc in dataset if doc["reference_code"] == ref][0]
            for gen in generations[i]:
                is_correct = test.test(gen)
                if is_correct:
                    num_correct += 1
                    break
        return {f"pass@{len(generations[0])} accuracy": num_correct / len(references)}
