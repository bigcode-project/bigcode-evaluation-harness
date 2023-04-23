import glob
import logging
import os
import pdb
import tempfile
from collections import defaultdict
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from lm_eval.tasks.custom_metrics.pie_perf_metric.sandbox import run_code_on_inputs

logging.basicConfig(level=logging.CRITICAL)

lang2file_ending = {
    "python": "py",
    "cpp": "cpp"
}

# path to test cases
inputs_outputs_basepath = "lm_eval/tasks/custom_metrics/pie_perf_metric/public_test_cases"
output_report_file_path = "codegen_1shot.jsonl.report"


def compute(generations, references, dataset):
    merged = read_inputs_and_prepare(generations, references, inputs=dataset['input'])
    problem_id_to_ground_truths, output_code_location, temp_dir = write_programs_read_ground_truth(merged)

    # run programs
    lang_file_ending = lang2file_ending["python"]
    tag_to_path = [
        ("input", f"_slow.{lang_file_ending}"),
        ("reference", f"_reference.{lang_file_ending}"),
    ]

    # check if there are multiple generations per input
    is_multigen = isinstance(merged["generated_answers"].iloc[0], list)
    if is_multigen:
        num_generations = len(merged["generated_answers"].iloc[0])
        tag_to_path.extend(
            [(f"{'generated_answers'}_{i}", f"_maybe_faster_{i}.{lang_file_ending}") for i
             in range(num_generations)])
    else:
        tag_to_path.append(("generated_answers", f"_maybe_faster_0.{lang_file_ending}"))

    results = run_programs(merged, problem_id_to_ground_truths, output_code_location, tag_to_path)

    if is_multigen:
        results = get_best_generation_per_submission(results, gen_col="generated_answers")

    metrics = generate_metric_report(merged, results, gen_col="generated_answers")

    if isinstance(temp_dir, tempfile.TemporaryDirectory):
        temp_dir.cleanup()

    return metrics


def read_inputs_and_prepare(generations, references, inputs: pd.Series) -> pd.DataFrame:
    """Reads the model generated output, the reference, joins them, and returns a dataframe with the merged data."""

    # gen_df = pd.read_json("generations.json")  # loading the model generations from jsonl to dataframe
    codes = [[subarr[0] for subarr in subsubarr] for subsubarr in generations]
    problem_ids = [[subarr[1] for subarr in subsubarr] for subsubarr in generations]
    submission_ids = [[subarr[2] for subarr in subsubarr] for subsubarr in generations]

    gen_df = pd.DataFrame({
        'generated_answers': codes,
        'problem_id': [i[0] for i in problem_ids],
        'submission_id_v0': [i[0] for i in submission_ids],
    })

    question_sep = "# slower version:"
    answer_sep = "# optimized version of the same code:"

    def process(x):
        x = x.replace("\n\n\n\n\n", "")
        x = x.split(question_sep)[-1].split(answer_sep)[0].strip()
        return x

    inputs = pd.Series(inputs.copy()).apply(process)
    gen_df['input'] = inputs

    gen_df["slower_program"] = gen_df["input"].apply(lambda x: x.strip())

    gen_df['target'] = pd.Series(references)
    merged = gen_df

    # if the generated code is a list, then we have multiple generations per input.
    # we add one column per generation
    if isinstance(merged["generated_answers"].iloc[0], list):
        num_generations = len(merged["generated_answers"].iloc[0])
        for i in range(num_generations):
            merged[f"{'generated_answers'}_{i}"] = merged["generated_answers"].apply(lambda x: x[i])
    return merged


def write_programs_read_ground_truth(merged: pd.DataFrame):
    # Writes all the programs to a temp directory, load ground truth
    # we don't want to do I/O repeatedly as it adds to the variance

    problem_id_to_ground_truths = defaultdict(list)
    temp_dir = tempfile.TemporaryDirectory()
    output_code_location = temp_dir.name

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="writing programs"):
        problem_id = row["problem_id"]

        # read the ground truth

        if problem_id not in problem_id_to_ground_truths:
            num_test_cases = len(
                glob.glob(f"{inputs_outputs_basepath}/{problem_id}/output*.txt")
            )
            assert (
                    num_test_cases > 0
            ), f"{inputs_outputs_basepath}/{problem_id} has no ground truth files!"
            for i in range(num_test_cases):
                with open(f"{inputs_outputs_basepath}/{problem_id}/output.{i}.txt") as f:
                    problem_id_to_ground_truths[problem_id].append(f.read().strip() + "\n")

        # write both generated and reference programs to the temp directory

        lang_file_ending = lang2file_ending["python"]
        submission_id_v0 = row["submission_id_v0"]
        with open(
                os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_slow.{lang_file_ending}"), "w"
        ) as f:
            ## This change in order to keep the comments out
            f.write(row["slower_program"])
            # f.write(row[cfg.slow_code_col].strip())

        # to deal with the case where there are multiple generated programs
        generated_programs = row["generated_answers"]

        if isinstance(generated_programs, str):
            generated_programs = [generated_programs]

        for i, generated_program in enumerate(generated_programs):
            with open(
                    os.path.join(output_code_location,
                                 f"{submission_id_v0}_{problem_id}_maybe_faster_{i}.{lang_file_ending}"), "w") as f:
                f.write(generated_program.strip())
        with open(
                os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_reference.{lang_file_ending}"), "w"
        ) as f:
            f.write(row["target"].strip())

    return problem_id_to_ground_truths, output_code_location, temp_dir


def run_programs(merged, problem_id_to_ground_truths: Dict, output_code_location: str, tag_to_path):
    results = dict()
    # NOTE: every row has a unique submission_id_v0, so we can use that as the submission_id
    # This is because for three submissions A, B, C, we create two pairs (A, B) and (B, C).
    # If we change this to also include (A, C), then we need to change the logic here. The following
    # assert checks that this is the case.
    assert len(merged["submission_id_v0"].unique()) == len(
        merged
    ), f"Every row should have a unique submission_id_v0. This is not the case: number of unique submission_id_v0: {len(merged['submission_id_v0'].unique())}, number of rows: {len(merged)}"

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="running programs"):
        problem_id = row["problem_id"]
        submission_id_v0 = row["submission_id_v0"]
        unit_test_data_basepath = f"{inputs_outputs_basepath}/{problem_id}"
        try:
            problem_execution_stats = dict()
            # run the generated program (maybe faster), input program (slower), and reference program (definitely
            # faster)
            for (tag, suffix) in tag_to_path:
                code_path = os.path.join(
                    output_code_location, f"{submission_id_v0}_{problem_id}{suffix}"
                )

                logging.info(
                    f"running {tag} program for problem {problem_id}, submission {submission_id_v0}"
                )

                avg_time, std_time, avg_acc = run_code_on_inputs(  # type: ignore
                    language="python",
                    code_path=code_path,
                    ground_truths=problem_id_to_ground_truths[problem_id],
                    unit_test_data_basepath=unit_test_data_basepath,
                    num_runs_per_test_case=25,
                    ignore_first_k=1,
                    max_seconds_per_run=10,
                    cpu_number=0,
                    cflags=None,
                    return_if_acc_below=None,
                )

                problem_execution_stats.update(
                    {
                        f"{tag}_time_mean": avg_time,
                        f"{tag}_time_std": std_time,
                        f"{tag}_acc": avg_acc,
                    }
                )
            results[submission_id_v0] = problem_execution_stats

        except Exception as e:
            tmp = dict()
            for tag, suffix in tag_to_path:
                tmp[f"{tag}_time_mean"] = np.nan
                tmp[f"{tag}_time_std"] = np.nan
                tmp[f"{tag}_acc"] = 0.0
            results[submission_id_v0] = tmp
            continue
    return results


def get_best_generation_per_submission(results: Dict, gen_col: str):
    best_per_sub = dict()
    for submission_id_v0, result_dict in results.items():
        gen_op_times = [(k, v) for k, v in result_dict.items() if gen_col in k and "time_mean" in k]
        gen_op_times = sorted(gen_op_times, key=lambda x: x[1])

        # itearte and find the first generation that is correct
        for gen_op_time in gen_op_times:
            if result_dict[f"{gen_op_time[0].replace('_time_mean', '')}_acc"] == 1.0:
                gen_op_times = [gen_op_time]
                break
        # find out which generation is the best
        try:
            best_gen_key = gen_op_times[0][0].replace("_time_mean", "")
            best_per_sub[submission_id_v0] = result_dict
            best_per_sub[submission_id_v0][f"{gen_col}_time_mean"] = gen_op_times[0][1]
            best_per_sub[submission_id_v0][f"{gen_col}_time_std"] = result_dict[f"{best_gen_key}_time_std"]
            best_per_sub[submission_id_v0][f"{gen_col}_acc"] = result_dict[f"{best_gen_key}_acc"]
        except IndexError:
            pdb.set_trace()

    return best_per_sub


def generate_metric_report(merged, results, gen_col: str):
    report_rows = []
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        submission_id_v0 = row["submission_id_v0"]

        if submission_id_v0 not in results:
            continue

        report_row = row.to_dict()

        report_row.update(results[submission_id_v0])
        report_rows.append(report_row)

    assert len(results) == len(report_rows)
    logging.info(f"Writing report to {output_report_file_path} with {len(report_rows)} rows")
    run_metrics = pd.DataFrame(report_rows)
    run_metrics.to_json(output_report_file_path, orient="records", lines=True)

    run_metrics = run_metrics[
        (run_metrics[f"{gen_col}_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)
        ]
    if run_metrics.empty:
        return {}

    # calculating metrics
    metrics = {}

    metrics['opt'] = len(
        run_metrics[run_metrics['generated_answers_time_mean'] < run_metrics['input_time_mean']]) * 100 / 1000
    metrics['sp'] = (run_metrics['input_time_mean'] / run_metrics['generated_answers_time_mean']).mean()

    efficient_generation_report = run_metrics[
        run_metrics['generated_answers_time_mean'] < run_metrics['input_time_mean']]
    metrics['rtr'] = ((efficient_generation_report['input_time_mean'] - efficient_generation_report[
        'generated_answers_time_mean']) * 100 / efficient_generation_report['input_time_mean']).mean()

    return metrics
