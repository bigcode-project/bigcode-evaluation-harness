import json
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_tokens, tokenizer_path):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_auth_token=True
        )
        self.samples = []  # Should be set in initialize()

        self.initialize(self.tokenizer)

    def initialize(self, tokenizer):

        all_samples = []
        skipped_problems = []

        all_samples_dict = {}  # Mapping from question_fname to list of samples
        count = 0
        for idx in tqdm(range(len(self.dataset))):
            sample = self.dataset[idx]
            # question
            question_str = sample["question"]
            # solutions
            try:
                solutions = json.loads(sample["solutions"])
            except ValueError:
                skipped_problems.append(idx)
                continue
            # starter code
            starter_code = (
                "" if len(sample["starter_code"]) == 0 else sample["starter_code"]
            )
            try:
                input_outpout = json.loads(sample["input_output"])
                fn_name = (
                    None
                    if not input_outpout.get("fn_name")
                    else input_outpout["fn_name"]
                )
            except ValueError:
                fn_name = None

            answer_type = (
                "\nUse Standard Input format\n"
                if not fn_name
                else "\nUse Call-Based format\n"
            )

            # Read all the solutions
            for solution in solutions:
                sample = (question_str, starter_code, solution, answer_type)
                # remove samples with long questions
                q_str = (
                    "\nQUESTION:\n"
                    + question_str
                    + "\n"
                    + starter_code
                    + "\n"
                    + answer_type
                    + "\nANSWER:\n"
                )
                if len(tokenizer(q_str)["input_ids"]) >= self.max_tokens:
                    count += 1
                    continue
                all_samples.append(sample)
                if question_str in all_samples_dict:
                    all_samples_dict[question_str].append(sample)
                else:
                    all_samples_dict[question_str] = [sample]

        print(f"Loaded {len(all_samples)} samples")
        print(f"Skipped {len(skipped_problems)} problems because no solution was found")
        print(f"Skipped {count} problems because the prompt was too long")
        self.samples = all_samples
        self.samples_dict = all_samples_dict

    def __len__(self):
        return len(self.samples)

    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the
        self.question_prefix. These will be added later and the total input will be
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = []

        curr_q, curr_s, curr_a, curr_q_prefix = self.samples[idx]

        while curr_num_tokens < self.max_tokens:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            # TODO change to one tokenizer call
            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))

            curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples)

        return curr_samples

    def __getitem__(self, idx):

        raw_samples = self.pack_samples(idx)
        output_samples = sample_gpt_task(
            raw_samples,
            max_tokens=self.max_tokens,
            tokenizer=self.tokenizer,
        )
        return output_samples


def sample_gpt_task(raw_samples, max_tokens, tokenizer):
    """
    Create the true sample used for the GPT task
    """

    input_ids = []
    label_ids = []

    for q_str, s_str, a_str, answer_type in raw_samples:

        # Loss is not calculated on this
        q_str = (
            "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"
        )

        question_token_ids = tokenizer(q_str)["input_ids"]
        answer_token_ids = tokenizer(a_str)["input_ids"] + [tokenizer.eos_token_id]

        input_ids.extend(question_token_ids + answer_token_ids)
        # labels must be of same size as inputs, -100 to ignore first tokens
        label_ids.extend([-100] * len(question_token_ids))
        label_ids.extend(answer_token_ids)

    # Sanity check
    assert len(input_ids) == len(label_ids)

    # Cut off the excess
    input_ids = input_ids[:max_tokens]
    label_ids = label_ids[:max_tokens]

    # TODO replace with a simple HF function/datacollator ?
    return {
        "input_ids": torch.LongTensor(input_ids),
        "labels": torch.LongTensor(label_ids),
    }


if __name__ == "__main__":
    import json

    from datasets import load_dataset

    # Do sanity checking
    dataset = load_dataset("codeparrot/apps", split="train")
    dataset.shuffle(seed=0)

    tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
    dataset = APPSBaseDataset(
        dataset, max_tokens=1024, tokenizer_path="codeparrot/codeparrot-small"
    )
    print("example sample of APPSBaseDataset:")
    example = dataset[0]
    labels = example["labels"]
    labels[labels == -100] = tokenizer.eos_token_id
    print(f"input ids {'-' * 10}:\n {tokenizer.decode(example['input_ids'])}")
    print(f"labels {'-' * 10}:\n {tokenizer.decode(labels)}")
