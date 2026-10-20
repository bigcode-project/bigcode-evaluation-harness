# Bayes@N for HumanEval and MBPP

Use `humaneval-bayes-at-n`, `humaneval-unstripped-bayes-at-n`, or
`mbpp-bayes-at-n` to report the posterior mean and posterior standard deviation
of average problem success probability alongside pass@k. These task variants
keep the original prompts, references, postprocessing and execution checks.

For example, evaluate 16 previously sampled programs per problem:

```sh
accelerate launch main.py --tasks humaneval-bayes-at-n \
    --load_generations_path generations.json --n_samples 16 \
    --allow_code_execution --metric_output_path evaluation_results.json
```

The generation file uses the existing list-of-lists format, in dataset order.
Set `--n_samples` to the intended number of retained programs per problem.
The normal generation path can use the same task names. Use independent
stochastic generations from a fixed model and sampling configuration.

Each task result gains two fields on the [0, 1] scale: `bayes_at_n` is the
posterior mean, and `bayes_at_n_std` is its posterior standard deviation.
For two problems with outcomes `[[0, 0], [1, 1]]`, they are `0.5` and about
`0.136931`. The original task names continue to report only their existing
metrics.

This implements the binary, uniform-prior case of
[Bayes@N](https://arxiv.org/abs/2510.04265), Algorithm 1 and Appendix A. Each
problem has its own Beta(1, 1) prior. If `c` of `N` programs pass the entire test
reference, its posterior is Beta(`c + 1`, `N - c + 1`). The corpus mean averages
problem posterior means. The corpus SD is `sqrt(sum(problem variances)) / M`,
where `M` is the number of selected problems. SD measures uncertainty about the
latent success probability of this fixed problem set, rather than variation
across benchmark problems or uncertainty in pass@k.

Every selected problem needs the same positive number of programs and a test
reference. Invalid or incomplete inputs raise `ValueError` before execution;
missing or duplicate execution verdicts also fail. Programs that fail tests or
time out are failures under the existing evaluator's grading rules. Historical
observations, categorical rubrics and credible intervals are outside this
integration's scope.

For evaluation-only files, use `--limit_start 0`. The existing loader does not
apply an offset to saved generation rows, so these variants reject a nonzero
offset to avoid pairing programs with the wrong references. Device-padding
candidates are cropped to `--n_samples` by the existing generation pipeline.
Intermediate checkpoints are unsupported for these variants: the existing
resume path compacts missing rows and cannot preserve their question identity.
Use a complete saved-generation file instead.
The standalone `compute_code_eval(..., bayes_at_n=True, expected_n=N)` API also
supports the calculation; if `expected_n` is omitted, it infers a fixed N from
the first row. Callers must retain every intended problem. A standalone matrix
cannot reveal a problem omitted entirely.
