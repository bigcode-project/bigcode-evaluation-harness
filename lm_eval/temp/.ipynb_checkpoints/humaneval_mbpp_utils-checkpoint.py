import re
from tqdm import tqdm

from transformers import pipeline, StoppingCriteria, StoppingCriteriaList


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(EOF_STRINGS), string)[0].rstrip()


def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
    return [first_block(code_gen["generated_text"][len(prompt) :]) for code_gen in code_gens]


def make_generations(model, tokenizer, dataset, args, num_tasks=None):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
    }

    # Generate completions for evaluation set
    print("Starting code generation")
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    generations, references = [], []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = dataset[task]["prompt"].strip()
        gen_kwargs["stopping_criteria"][0].start_length = len(tokenizer(prompt)["input_ids"])
        for batch in range(args.n_samples // args.batch_size):
            task_generations.extend(complete_code(pipe, prompt, num_completions=args.batch_size, **gen_kwargs))
        generations.append([prompt + gen for gen in task_generations])
        test_func = dataset[task]["test"]
        entry_point = f"check({dataset[task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)
    return generations, references