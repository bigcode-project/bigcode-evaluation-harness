from copy import deepcopy
import gc
import pickle
import time

import io
import os
import sys
import resource
import platform
import contextlib

import numpy as np

def calc_exec_time(ts): # Hodges--Lehmann estimator
    ts = np.array(ts) / 2.
    ts = ts[None, :] + ts[:, None]
    ts = ts[np.tril_indices_from(ts)]
    return np.median(ts)

def calc_eff(elapsed, ref, timeout):
    return max(0., timeout - elapsed) / (timeout - ref)

def calc_eff_at_k(e, k): # numerically stable implementation
    n = len(e)
    lbd = [k / n]
    k_ = k - 1
    for r in range(n - 1, k_, -1):
        lbd.append(lbd[-1] * (1 - k_ / r))
    lbd = np.flip(lbd)
    e = np.sort(e)[k_ :]
    return (lbd * e).sum()

def calc_pass_at_k(n, c, k): # from the HumanEval paper
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

class Test: # a test case
    def __init__(self, input = None, answer = None, ref = None):
        self.input = input
        self.answer = answer
        self.ref = ref # reference execution time

class Refs: # references for efficiency evaluation
    def __init__(self, tests, hardness):
        neg_inf = float('-inf')
        self.refs = [neg_inf] * len(hardness)
        self.ref_max = neg_inf
        self.lid = None
        self.cid = None
        # finds the longest reference execution time for calibration
        for j, (size, tests_j) in enumerate(tests):
            if hardness[j]:
                for k, test in enumerate(tests_j):
                    if self.refs[j] < test.ref:
                        self.refs[j] = test.ref
                        if self.ref_max < test.ref:
                            self.ref_max = test.ref
                            self.lid = j
                            self.cid = k

class EnamUnpickler(pickle.Unpickler):
    CLS_DICT = {'enam.evaluate.Test': Test, 'enam.evaluate.Refs': Refs}
    def find_class(self, module, name):
        if module in self.CLS_DICT:
            return self.CLS_DICT[module]
        else:
            return super().find_class(module, name)

TPL_RUN = '''%s
%s
__t0 = time.time()
__output = %s(*__input)
__t1 = time.time()
''' # % (prompt, code, entry_point) # this should work no matter code includes prompt or not
TPL_TEST = '''%s
    pass
%s
__accepted = __check(__input, __answer, __output)
''' # % (prompt, checker)

def evaluate_one(code, problem, tests, refs, k, hardness, n_reps, memory_giga, timeout_factor, tolerence_sec, time_correction):
    timeout = timeout_factor * refs.ref_max
    memory_bytes = memory_giga * (1024 ** 3)
    effs = []
    elapsed_list = []
    for j, (size, tests_j) in enumerate(tests):
        n_reps_j = n_reps[j]
        level_elapsed = []
        level_break = False
        for k, test in enumerate(tests_j):
            elapsed = [None for rep in range(n_reps_j)]
            for rep in range(n_reps):
                scope = dict(time = time, input = None, print = None, __input = deepcopy(test.input)) # in case that the code modifies the input
                try:
                    unsafe_timed_execute(TPL_RUN % (problem.prompt, code, problem.entry_point), scope, memory_bytes, timeout + tolerence_sec)
                    scope['__input'] = test.input
                    scope['__answer'] = test.answer # to prevent the code reading the answer
                    unsafe_execute(TPL_TEST % (problem.prompt, problem.checker), scope) # assuming that the checker does not modify the input
                except TimeoutException as e:
                    level_break = True
                    break
                except MemoryError as e:
                    level_break = True
                    break
                except OverflowError as e:
                    level_break = True
                    break
                except KeyboardInterrupt as e:
                    raise e
                except BaseException as e:
                    return False, self.zero_effs(), elapsed_list
                else:
                    if '__accepted' in scope and scope['__accepted']:
                        elapsed[rep] = scope['__t1'] - scope['__t0']
                    else:
                        return False, self.zero_effs(), elapsed_list
            if level_break:
                break
            else:
                level_elapsed.append(calc_exec_time(elapsed).item() * time_correction)
        elapsed_list.append(level_elapsed)
        if level_break:
            break
        else:
            effs.append(calc_eff(elapsed = max(level_elapsed), ref = refs.refs[j], timeout = timeout))
    if j == 0 and level_break:
        return False, self.zero_effs(), elapsed_list
    for j in range(len(effs), self.n_levels):
        effs.append(0.)
    return True, effs, elapsed_list

def get_time_correction(problem, tests, refs, n_reps): # computes the calibration factor of of execution time
    j = refs.lid
    k = refs.cid
    test = tests[j][-1][k]
    n_reps_j = n_reps[j]
    elapsed = [None for rep in range(n_reps_j)]
    for rep in range(n_reps_j):
        scope = dict(time = time, __input = deepcopy(test.input)) # in case that the code modifies the input
        unsafe_execute(TPL_RUN % (problem.prompt, problem.reference_solution, problem.entry_point), scope) # assuming that the reference solution is error-free
        elapsed[rep] = scope['__t1'] - scope['__t0']
    elapsed = calc_exec_time(elapsed).item()
    return refs.ref_max / elapsed

def evaluate_all(problems, codes, tests, refs, k, hardness, n_reps, memory_giga, timeout_factor, tolerence_sec):
    if isinstance(k, int):
        k = [k]
    min_codes = min(len(codes_i) for codes_i in codes)
    k = sorted({k_ for k_ in k if k_ <= min_codes})
    passes = [[] for k_ in k]
    effs = [[] for k_ in k]
    gc.collect()
    for problem, codes_i, tests_i, refs_i in zip(problems, codes, tests, refs):
        time_correction = get_time_correction(problem = problem, tests = tests_i, refs = refs_i, n_reps = n_reps)
        n_levels = len(tests_i)
        problem_passes = []
        problem_effs = []
        for code in codes_i:
            passed, code_effs, code_elapsed = evaluate_one(
                code = code, problem = problem, tests = tests_i, refs = refs_i,
                k = k, hardness = hardness, n_reps = n_reps, memory_giga = memory_giga,
                timeout_factor = timeout_factor, tolerence_sec = tolerence_sec, time_correction = time_correction)
            problem_passes.append(passed)
            problem_effs.append(code_effs)
        for j, k_ in enumerate(k):
            passes[j].append(calc_pass_at_k(n = len(problem_passes), c = sum(problem_passes), k = k_))
            effs[j].append(calc_eff_at_k(e = np.average(problem_effs, axis = 1, weights = hardness), k = k_))
    metrics = dict()
    for k_, pass_k in zip(k, passes):
        metrics[f'pass@{k_}'] = np.mean(pass_k).item()
    for k_, eff_k in zip(k, effs):
        metrics[f'eff@{k_}'] = np.mean(eff_k).item()
    return metrics

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

"""The following functions are adapted from code_eval (@link https://huggingface.co/spaces/evaluate-metric/code_eval)"""

def get_memory_usage():
    return sys.getsizeof(sys.modules[__name__])

@contextlib.contextmanager
def set_memory_limit(maximum_memory_bytes = None):
    try:
        if maximum_memory_bytes is not None:
            _not_darwin = (not platform.uname().system == "Darwin")
            _rlimit_as = resource.getrlimit(resource.RLIMIT_AS)
            _rlimit_data = resource.getrlimit(resource.RLIMIT_DATA)
            if _not_darwin:
                _rlimit_stack = resource.getrlimit(resource.RLIMIT_STACK)
            memory_limit = int(get_memory_usage() + maximum_memory_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, _rlimit_as[-1]))
            resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, _rlimit_data[-1]))
            if _not_darwin:
                resource.setrlimit(resource.RLIMIT_STACK, (memory_limit, _rlimit_stack[-1]))
        yield
    finally:
        if maximum_memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, _rlimit_as)
            resource.setrlimit(resource.RLIMIT_DATA, _rlimit_data)
            if _not_darwin:
                resource.setrlimit(resource.RLIMIT_STACK, _rlimit_stack)

class TimeoutException(Exception):
    pass

def timeout_signal_handler(signum, frame):
    raise TimeoutException("Timed out!")

@contextlib.contextmanager
def set_time_limit(seconds):
    import signal
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, timeout_signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise OSError
    def readline(self, *args, **kwargs):
        raise OSError
    def readlines(self, *args, **kwargs):
        raise OSError
    def readable(self, *args, **kwargs):
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

@contextlib.contextmanager
def create_tempdir():
    import tempfile
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

@contextlib.contextmanager
def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    with create_tempdir():
        with swallow_io():
            try:

                import faulthandler

                faulthandler.disable()

                import builtins, os, shutil, subprocess

                os.environ["OMP_NUM_THREADS"] = "1"

                _keys = dict(
                    builtins = ('exit', 'quit'),
                    os = ('kill', 'system', 'putenv', 'remove', 'removedirs', 'rmdir', 'fchdir', 'setuid', 'fork', 'forkpty', 'killpg', 'rename', 'renames', 'truncate', 'replace', 'unlink', 'fchmod', 'fchown', 'chmod', 'chown', 'chroot', 'lchflags', 'lchmod', 'lchown', 'getcwd', 'chdir'),
                    shutil = ('rmtree', 'move', 'chown'),
                    subprocess = ('Popen',),
                )
                _baks = dict()
                for lib, keys in _keys.items():
                    obj = locals()[lib]
                    _bak = dict()
                    for key in keys:
                        if hasattr(obj, key):
                            _bak[key] = getattr(obj, key)
                    _baks[lib] = _bak

                #__builtins__["help"] = None

                yield
            finally:
                for lib, keys in _keys.items():
                    obj = locals()[lib]
                    for key, val in _baks[lib].items():
                        setattr(obj, key, val)

def unsafe_execute(program: str, exec_globals: dict):
    try:
        gc_bak = gc.isenabled()
        gc.disable()
        with reliability_guard():
            exec(program, exec_globals)
    finally:
        if gc_bak:
            gc.enable()

def unsafe_timed_execute(program: str, exec_globals: dict, maximum_memory_bytes: float, time_limit_seconds: float):
    try:
        gc_bak = gc.isenabled()
        gc.disable()
        with reliability_guard():
            with set_memory_limit(maximum_memory_bytes):
                with set_time_limit(time_limit_seconds):
                    exec(program, exec_globals)
    finally:
        if gc_bak:
            gc.enable()
