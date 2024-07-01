# TODO: eff@k

def evaluate_all(generations, references, k, hardness, n_reps, memory_giga, timeout_factor, tolerence_sec):
    # TODO

def might_catch_timeout_signal():
    # TODO

might_catch_timeout_signal.WARNING = """\
We have detected that the generated code samples use `try ... except` within a loop, which might catch \
our timeout signal and cause a dead loop. Since resolving this rare issue via `multiprocessing` would \
significantly slow down the evaluation process for our large-scale inputs, we have decided not to resolve \
this issue. If this issue does happen, please consider removing the corresponding code samples."""
