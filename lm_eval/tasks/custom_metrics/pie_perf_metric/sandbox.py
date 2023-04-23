# utilities and code that handles actual execution of programs
import glob
import logging
import os
import pathlib
import resource
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import psutil

# disable logging from psutil
logging.getLogger("psutil").setLevel(logging.WARNING)

# disable logging from resource
logging.getLogger("resource").setLevel(logging.WARNING)

# disable logging from subprocess
logging.getLogger("subprocess").setLevel(logging.WARNING)

logging.basicConfig(level=logging.CRITICAL)

DEBUG = True


def is_linux():
    from sys import platform
    if platform == "linux" or platform == "linux2":
        return True
    else:
        return False


def run_python_code_on_inputs(
        code_path: str,
        unit_test_data_basepath: str,
        num_runs_per_test_case: int,
        ignore_first_k: int,
        max_seconds_per_run: int,
        ground_truths: List[str] = None,  # type: ignore
        num_test_cases: int = None,  # type: ignore
        cpu_number: int = 1,  # which CPU to run the code on, counting begins from 1
        return_per_trial_times: bool = False,
        python_bin: str = "python",
        return_dict: bool = False,
        cflags: str = None,  # type: ignore
        return_if_acc_below: float = None,  # ignore  if the accuracy is below this value, then return
) -> Union[Tuple[float, float, float], Tuple[float, float, float, List[List[float]]], Dict]:
    """
    Run the given code on the inputs for the given problem_id, and returns (avg_time, std_time, avg_acc).
    The inputs are sourced from the unit test data, where a number of files of the form: {input,output}.{0, 1, 2}.txt are present.


    NOTE: It is optional to pass ground_truths. If they are not passed, then the accuracy will not be computed.


    """

    if num_test_cases is None:
        num_test_cases = len(ground_truths)

    times_millisec, accs = [], []
    per_trial_times = []
    for test_case_idx in range(num_test_cases):
        if is_linux():
            cmd = (
                f"taskset --cpu-list {cpu_number} {python_bin} {code_path}"  # taskset 00 python code.py
            )
        else:
            cmd = f"{python_bin} {code_path}"
        subprocess_args = shlex.split(cmd)
        input_file_path = f"{unit_test_data_basepath}/input.{test_case_idx}.txt"
        # logging.info(f"Running command: {cmd} < {input_file_path}")
        _per_trial_times = []
        for trial_idx in range(num_runs_per_test_case):
            try:
                time_start = time.time()
                output = run_cmd_for_time_eval(
                    subprocess_args,
                    input_file_path=input_file_path,
                    timeout_seconds=max_seconds_per_run,
                )
                time_taken = time.time() - time_start
                _per_trial_times.append(time_taken)
                if output is None:
                    return (np.nan, np.nan, 0)
                    # timeout: since we have a generous timeout, this should not happen

                if trial_idx >= ignore_first_k:
                    times_millisec.append(time_taken * 1000)
                    if ground_truths is not None:
                        accuracy = get_accuracy(output, ground_truths[test_case_idx])
                        if return_if_acc_below is not None and accuracy < return_if_acc_below:
                            logging.info(f"Accuracy {accuracy} below {return_if_acc_below}. Returning.")
                            return (time_taken, 0, accuracy)
                        accs.append(accuracy)

            except Exception as e:
                logging.warning("Error", e)
                # no point in repeating the test for this problem. If something went wrong, it will go wrong again
                return (np.nan, np.nan, 0)

        per_trial_times.append(_per_trial_times)

    times_millisec, accs = np.array(times_millisec), np.array(accs)
    if return_per_trial_times and ground_truths is None:
        return per_trial_times  # type: ignore
    if return_dict:
        return {
            "avg_time": np.mean(times_millisec),
            "std_time": np.std(times_millisec),
            "avg_acc": np.mean(accs),
        }
    else:
        return np.mean(times_millisec), np.std(times_millisec), np.mean(accs)  # type: ignore


# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB


# from https://gist.github.com/s3rvac/f97d6cbdfdb15c0a32e7e941f7f4a3fa
def limit_virtual_memory():
    # The tuple below is of the form (soft limit, hard limit). Limit only
    # the soft part so that the limit can be increased later (setting also
    # the hard limit would prevent that).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    if is_linux():
        resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY * 10))
    else:
        pass


def run_cmd_for_time_eval(args, input_file_path: str, timeout_seconds: int = 3) -> Union[str, None]:
    def _kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            # logging.info(f"Killing {proc}")
            proc.kill()
        # logging.info(f"Killing {process}")
        process.kill()

    try:
        with open(input_file_path, "r") as f:
            proc = subprocess.Popen(
                args,
                stdin=f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=limit_virtual_memory,
            )

            output = proc.communicate(timeout=timeout_seconds)[0]
            # if proc.returncode != 0:
            #     with open("error.txt", "a") as f:
            #         f.write(shlex.join(args) + "<" + input_file_path + "\n")
            #     _kill(proc.pid)
            return output.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        # print(f"Timeout for {args}")
        _kill(proc.pid)  # type: ignore
        return None


def get_accuracy(output: str, ground_truth: str) -> float:
    """
    Compare the output of the code with the ground truth.
    """
    num_correct = 0
    ground_truth_lines = ground_truth.strip().splitlines()
    output_truth_lines = output.strip().splitlines()
    for gen_output, ground_truth_output in zip(output_truth_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        num_correct += int(is_corr)

    return num_correct / len(ground_truth_lines)


def run_cmd_for_time_eval_n_times(
        cmd: str, input_path: str, n: int, timeout_seconds: int = 1
) -> List[float]:
    times = []
    for i in range(n):
        time_start = time.time()
        run_cmd_for_time_eval(cmd, input_path, timeout_seconds=timeout_seconds)
        time_taken = time.time() - time_start
        times.append(time_taken)
    return times


def test_python():
    import shutil
    from pprint import pprint
    slow_sum_code = """
def sum_n_numbers_slow(n: int) -> int:
    sum = 0
    for i in range(n + 1):
        sum += i
    print(sum)
if __name__ == "__main__":
    sum_n_numbers_slow(int(input()))
"""
    fast_sum_code = """
def sum_n_numbers_fast(n: int) -> int:
    print(n * (n + 1) / 2)

if __name__ == "__main__":
    sum_n_numbers_fast(int(input()))
"""

    fast_but_wrong_sum_code = """
def sum_n_numbers_fast(n: int) -> int:
    print(n * (n - 1) / 2)

if __name__ == "__main__":
    sum_n_numbers_fast(int(input()))
"""

    test_cases = {
        "slow": slow_sum_code,
        "fast": fast_sum_code,
        "fast_but_wrong": fast_but_wrong_sum_code,
    }
    ground_truths, temp_dir_name = write_test_inputs()

    results = {code_type: {} for code_type in test_cases}
    for (code_type, code) in test_cases.items():
        with open(f"{temp_dir_name}/{code_type}.py", "w") as f:
            f.write(code)
        code_type_results = run_python_code_on_inputs(  # type: ignore
            code_path=f"{temp_dir_name}/{code_type}.py",
            unit_test_data_basepath=temp_dir_name,
            num_runs_per_test_case=10,
            ignore_first_k=2,
            max_seconds_per_run=10,
            ground_truths=ground_truths,
            cpu_number=2,
            return_dict=True
        )
        results[code_type].update(code_type_results)  # type: ignore

    assert results["slow"]["avg_time"] > results["fast"]["avg_time"]
    assert results["fast"]["avg_acc"] == 1.0
    assert results["slow"]["avg_acc"] == 1.0
    assert results["fast_but_wrong"]["avg_acc"] == 0.0
    shutil.rmtree(temp_dir_name)
    print("Test passed! Results: ")
    pprint(results)


def make_temp_dir():
    import uuid
    temp_dir_name = f"/tmp/{uuid.uuid4()}"
    pathlib.Path(temp_dir_name).mkdir(parents=True, exist_ok=True)
    return temp_dir_name


def write_test_inputs(inputs=["10000", "1000000"]):
    # writes the inputs to a temporary directory.
    temp_dir_name = make_temp_dir()

    # create a file with the ground truths called inputs.{i}.txt
    for i, input_txt in enumerate(inputs):
        with open(f"{temp_dir_name}/input.{i}.txt", "w") as input_file:
            print(f"Wrote input # {i} to {input_file.name}")
            input_file.write(input_txt)
    ground_truths = [str(sum(range(int(i) + 1))) for i in inputs]

    return ground_truths, temp_dir_name


def compile_cpp_code(code_path: str, output_path: str = None, cflags: str = "") -> str:
    """_summary_

    Args:
        code_path (str): _description_
        output_path (str, optional): _description_
        cflags (str, optional): _description_

    Returns:
        str: _description_
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(code_path), "a.out")
    cmd = ["/usr/bin/g++", code_path, "-o", output_path] + shlex.split(cflags.replace('"', "").replace("'", ""))
    logging.info(f"Running command: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise Exception(
            f"Error compiling code: {code_path} with command: {' '.join(cmd)}, return code: {p.returncode}, stderr: {p.stderr.decode('utf-8')}")
    return output_path


def run_cpp_code_on_inputs(
        code_path: str,
        unit_test_data_basepath: str,
        num_runs_per_test_case: int,
        ignore_first_k: int,
        max_seconds_per_run: int,
        ground_truths: List[str] = None,  # type: ignore
        num_test_cases: int = None,  # type: ignore
        cpu_number: int = 1,  # which CPU to run the code on, counting begins from 1
        return_per_trial_times: bool = False,
        python_bin: str = "python",  # unused
        return_dict: bool = False,
        remove_code_after_run: bool = True,
        debug_stderr=sys.stderr,  # temporary for debugging purposes
        cflags: str = "--std=c++17 -O1",
        return_if_acc_below: float = 0.0,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, List[List[float]]], Dict]:
    """
    Run the given code on the inputs for the given problem_id, and returns (avg_time, std_time, avg_acc).
    The inputs are sourced from the unit test data, where a number of files of the form: {input,output}.{0, 1, 2}.txt are present.


    NOTE: It is optional to pass ground_truths. If they are not passed, then the accuracy will not be computed.


    """
    try:
        binary_output_path = compile_cpp_code(code_path, cflags=cflags)
    except Exception as e:
        logging.warning(f"Error: {e}")
        return (np.nan, np.nan, 0)

    if num_test_cases is None:
        num_test_cases = len(ground_truths)

    times_millisec, accs = [], []
    per_trial_times = []
    for test_case_idx in range(num_test_cases):
        if is_linux():
            cmd = (
                f"taskset --cpu-list {cpu_number} {binary_output_path}"  # taskset 00 python code.py
            )
        else:
            cmd = f"{binary_output_path}"
        subprocess_args = shlex.split(cmd)
        input_file_path = f"{unit_test_data_basepath}/input.{test_case_idx}.txt"
        _per_trial_times = []
        for trial_idx in range(num_runs_per_test_case):
            try:
                time_start = time.time()
                output = run_cmd_for_time_eval(
                    subprocess_args,
                    input_file_path=input_file_path,
                    timeout_seconds=max_seconds_per_run,
                )
                time_taken = time.time() - time_start
                _per_trial_times.append(time_taken)
                if output is None:
                    if remove_code_after_run:
                        os.remove(binary_output_path)
                    return (np.nan, np.nan, 0)
                    # timeout: since we have a generous timeout, this should not happen

                if trial_idx >= ignore_first_k:
                    times_millisec.append(time_taken * 1000)
                    if ground_truths is not None:
                        acc = get_accuracy(output, ground_truths[test_case_idx])
                        if acc < return_if_acc_below:
                            if remove_code_after_run:
                                os.remove(binary_output_path)
                            logging.info(f"Accuracy {acc} below {return_if_acc_below}. Returning.")
                            return (np.nan, np.nan, 0)
                        accs.append(acc)

            except Exception as e:
                logging.warning("Error", e)
                # no point in repeating the test for this problem. If something went wrong, it will go wrong again
                return (np.nan, np.nan, 0)

        per_trial_times.append(_per_trial_times)

    times_millisec, accs = np.array(times_millisec), np.array(accs)
    if return_per_trial_times and ground_truths is None:
        return per_trial_times  # type: ignore
    if return_dict:
        return {
            "avg_time": np.mean(times_millisec),
            "std_time": np.std(times_millisec),
            "avg_acc": np.mean(accs),
        }
    else:
        return np.mean(times_millisec), np.std(times_millisec), np.mean(accs)  # type: ignore


def test_cpp():
    import shutil
    from pprint import pprint
    slow_sum_code_path = "src/codenet_eval/cpp_examples/slow_num.cpp"
    fast_num_code_path = "src/codenet_eval/cpp_examples/fast_num.cpp"
    fast_but_wrong_code_path = "src/codenet_eval/cpp_examples/fast_but_wrong.cpp"
    test_cases = {
        "slow": slow_sum_code_path,
        "fast": fast_num_code_path,
        "fast_but_wrong": fast_but_wrong_code_path
    }
    ground_truths, temp_dir_name = write_test_inputs()
    results = {code_type: {} for code_type in test_cases}
    for (code_type, code_pth) in test_cases.items():
        code_type_results = run_cpp_code_on_inputs(  # type: ignore
            code_path=code_pth,
            unit_test_data_basepath=temp_dir_name,
            num_runs_per_test_case=10,
            ignore_first_k=2,
            max_seconds_per_run=10,
            ground_truths=ground_truths,
            cpu_number=2,
            return_dict=True
        )
        results[code_type].update(code_type_results)  # type: ignore

    assert results["slow"]["avg_time"] > results["fast"]["avg_time"]
    assert results["fast"]["avg_acc"] == 1.0
    assert results["slow"]["avg_acc"] == 1.0
    assert results["fast_but_wrong"]["avg_acc"] == 0.0
    shutil.rmtree(temp_dir_name)
    print("Test passed! Results: ")
    pprint(results)


def test_cpp_reference(number_to_test: int, path_to_ref: str, report_dir: str, test_case_path: str) -> None:
    """
    Takes the path to the reference file, and the path to the test cases,
    and it checks that all (input, output) pairs in the reference file
    can be compiled and run on the test cases and also ensures that the
    outputs are correct.

    The output file is used as an input for the evaluation script (to determine which examples to exclude)
    """
    import json
    from tqdm import tqdm
    import uuid

    def write_dict(d, fh):
        for k, v in d.items():
            fh.write(f"{'*' * 40}\n")
            fh.write(f"{'*' * 15} {k} {'*' * 15}\n")
            fh.write(f"{'*' * 40}\n\n\n")
            fh.write(str(v) + "\n\n\n")

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    with open(path_to_ref, "r") as f:
        lines = f.readlines()
        refs = [json.loads(line) for line in lines][:number_to_test]
    meta_results_dict = {
        "slow_compiled": 0,
        "slow_ran": 0,
        "fast_compiled": 0,
        "fast_ran": 0,
        "fast_is_faster": 0
    }
    pbar = tqdm(total=len(refs))
    all_results = {i: {} for i in range(len(refs))}

    for i, ref in enumerate(refs):
        problem_id = ref["problem_id"]
        problem_dir = os.path.join(report_dir, uuid.uuid4().hex)
        # problem_dir = os.path.join(report_dir, problem_id)
        if not os.path.exists(problem_dir):
            os.mkdir(path=problem_dir)
            print(f"Created directory {problem_dir}")
        slow_code = ref["input"]
        fast_code = ref["target"]
        # if "#include <iostream>" not in slow_code:
        #     slow_code = "#include <iostream> \n" + slow_code
        # if "#include <iostream>" not in fast_code:
        #     fast_code = "#include <iostream> \n" + fast_code
        slow_path = os.path.join(problem_dir, "slow.cpp")
        fast_path = os.path.join(problem_dir, "fast.cpp")
        with open(slow_path, "w") as f:
            f.write(slow_code)
        with open(fast_path, "w") as f:
            f.write(fast_code)
        test_cases = {
            "slow": slow_path,
            "fast": fast_path
        }

        ## ground truths
        ground_truths = []
        num_test_cases = len(
            glob.glob(f"{test_case_path}/{problem_id}/output*.txt")
        )
        assert (
                num_test_cases > 0
        ), f"{test_case_path}/{problem_id} has no ground truth files!"
        for j in range(num_test_cases):
            with open(f"{test_case_path}/{problem_id}/output.{j}.txt") as f:
                ground_truths.append(f.read().strip() + "\n")

        results = {code_type: {} for code_type in ["slow", "fast"]}
        results["problem_id"] = problem_id
        ## the debug file was part of the original code, but it adds some clunk to the
        ## run_cpp_code_on_inputs function

        # debug_file = os.path.join(problem_dir, "debug.txt")
        # with open(debug_file, "w") as debug_stderr:
        for (code_type, code_pth) in test_cases.items():

            code_type_results = run_cpp_code_on_inputs(  # type: ignore
                code_path=code_pth,
                unit_test_data_basepath=os.path.join(test_case_path, problem_id),
                num_runs_per_test_case=2,
                ignore_first_k=0,
                max_seconds_per_run=10,
                ground_truths=ground_truths,
                cpu_number=2,
                return_dict=True,
                remove_code_after_run=False,
                # debug_stderr=debug_stderr,
                cflags="--std=c++17 -O1"
            )

            compiled = False
            ran = False
            if not code_type_results:
                code_type_results = {
                    "avg_time": np.nan,
                    "std_time": np.nan,
                    "avg_acc": 0,
                }
            elif isinstance(code_type_results, tuple):
                code_type_results = {
                    "avg_time": np.nan,
                    "std_time": np.nan,
                    "avg_acc": 0,
                }
                compiled = True
            else:
                compiled = True
                ran = True
            code_type_results.update({"compiled": compiled, "ran": ran})
            results[code_type].update(code_type_results)  # type: ignore
            meta_results_dict[f"{code_type}_compiled"] += compiled
            meta_results_dict[f"{code_type}_ran"] += ran
            all_results[i][f"{code_type}_compiled"] = compiled
            all_results[i][f"{code_type}_ran"] = ran
            all_results[i][f"{code_type}_avg_time"] = code_type_results["avg_time"]
            all_results[i][f"{code_type}_std_time"] = code_type_results["std_time"]
            all_results[i][f"{code_type}_avg_acc"] = code_type_results["avg_acc"]
            all_results[i]["input"] = slow_code
            all_results[i]["target"] = fast_code

        if results["slow"].get("compiled") and results["fast"].get("compiled"):
            meta_results_dict["fast_is_faster"] += (results["fast"]["avg_time"] < results["slow"]["avg_time"])

        with open(f"{problem_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)
            print(f"Saved results to {problem_dir}/results.json")
        with open(f"{problem_dir}/ref.txt", "w") as f:
            write_dict(ref, f)

        pbar.update(1)
        pbar.set_description(
            f"Compiled {meta_results_dict['slow_compiled'] + meta_results_dict['fast_compiled']}/{(i + 1) * 2}, Ran {meta_results_dict['slow_ran'] + meta_results_dict['fast_ran']}/{(i + 1) * 2}, Fast is faster {meta_results_dict['fast_is_faster']}/{i + 1}")
    pbar.close()
    with open(f"{report_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        print(f"Saved results to {report_dir}/all_results.json")


def run_code_on_inputs(*args, **kwargs):
    language = kwargs.pop("language")
    if language == "python":
        return run_python_code_on_inputs(*args, **kwargs)
    elif language == "cpp":
        return run_cpp_code_on_inputs(*args, **kwargs)


def test():
    test_python()
    test_cpp()


if __name__ == "__main__":
    test()
