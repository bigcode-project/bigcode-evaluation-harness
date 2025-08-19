from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """
    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
        },
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )
    length_penalty: Optional[dict[str, int | float]] = field(
        default=None,
        metadata={"help": "A dictionary with length penalty options (for watsonx.ai)."}
    )
    max_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of generated tokens (for watsonx.ai)."}
    )
    min_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "Minimum number of generated tokens (for watsonx.ai)."}
    )
    stop_sequences: Optional[list[str]] = field(
        default=None, metadata={"help": "List of stop sequences (for watsonx.ai)."}
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "A float value of repetition penalty (for watsonx.ai)."}
    )
