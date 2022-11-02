from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """

    model_ckpt: Optional[str] = field(
        default="codeparrot/codeparrot",
        metadata={"help": "Model name in Hugging Face hub or its local path."},
    )
    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the generated code. For exmaple InCoder need prefix='<| file ext=.py |>\n'"
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
        default=200,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )
