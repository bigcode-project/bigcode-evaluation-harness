import json
from tqdm import tqdm
from transformers import pipeline

def generate_prompt(sample):
    """APPS prompts include a question along with some starter code and function name if they exist
    We also specify the type of the prompt, i.e. whether it is call-based or standard input"""

    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"] 
    except ValueError:
        fn_name = None 
    _input = "\nQUESTION:\n"
    _input += sample["question"]
    if starter_code:
        _input += starter_code
    if fn_name:
        _input += "\nUse Standard Input format"
    else:
        _input += "\nUse Call-Based format"
    
    _input += "\nANSWER:\n"
    return _input


def complete_code(pipe, prompt, num_completions=1, max_length=1024, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    try:
        code_gens = pipe(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs)
        return [code_gen["generated_text"][len(prompt):] for code_gen in code_gens]
    except IndexError:
        print("prompt is longer than the context size of the model, generation skipped")
        code_gens = ""
        return [""]


def make_generations(model, tokenizer, dataset, args, num_tasks=None):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    # Generate completions for evaluation set
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    print(f"ntasks is {n_tasks}")
    generations = []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = generate_prompt(dataset[task]).strip()
        task_generations.extend(complete_code(pipe, prompt, num_completions=args.n_samples, max_length=args.max_length, **gen_kwargs))
        generations.append([gen.replace(args.eos, "") for gen in task_generations])
    return generations