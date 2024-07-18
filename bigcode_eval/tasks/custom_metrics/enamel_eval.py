# TODO: eff@k


class Unpickler(pickle.Unpickler):
    CLS_DICT = {'': Test, '': Refs}
    def find_class(self, module, name):
        if module in self.CLS_DICT:
            return self.CLS_DICT[module]
        else:
            return super().find_class(module, name)


def evaluate_all(generations, references, k, hardness, n_reps, memory_giga, timeout_factor, tolerence_sec):
    # TODO


def might_catch_timeout_signal(generation, pattern_seq = ('    while ', '     try:')):
    i = 0
    for pattern in pattern_seq:
        i = generarion.find(pattern, i)
        if i == -1:
            return False
        i += len(pattern)
    return True

might_catch_timeout_signal.WARNING = """\
We have detected that the generated code samples use `try ... except` within a loop, which might catch \
our timeout signal and cause a dead loop. Since resolving this rare issue via `multiprocessing` would \
significantly slow down the evaluation process for our large-scale inputs, we have decided not to resolve \
this issue. If this issue does happen, please consider removing the corresponding code samples."""
