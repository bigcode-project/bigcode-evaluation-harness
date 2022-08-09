from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """

    model_ckpt: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Model name in Hugging Face hub or its local path."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    num_tasks_humaneval: Optional[int] = field(
        default=None,
        metadata={"help": "The number of human-eval tasks to run. If not specified all tasks are evaluated."},
    )
    num_tasks_apps: Optional[int] = field(
        default=None,
        metadata={"help": "The number of APPS tasks to run. If not specified all tasks are evaluated."},
    )
    num_tasks_mbpp: Optional[int] = field(
        default=None,
        metadata={"help": "The number of APPS tasks to run. If not specified all tasks are evaluated."},
    )
    level_apps: Optional[str] = field(
        default="all",
        metadata={"help": "The difficulty level to use for APPS, among introductory, interview, competition and all."},
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Sample from the language model's output distribution."}
    )
    temperature: Optional[float] = field(default=0.2, metadata={"help": "Sampling temperature used for generation."})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of newly generated tokens, used in humaneval."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "Maximum length of prompt+generated sequence, used in APPS and MBPP"})
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."})
    n_samples: Optional[int] = field(
        default=200, metadata={"help": "Number of completions to generate for each sample."}
    )
    eos : Optional[str] = field(default="<|endoftext|>", metadata={"help": "end of sentence token."})
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed used for evaluation."})