# SPDX-License-Identifier: Apache-2.0
"""Binary Bayes@N moments from https://arxiv.org/abs/2510.04265."""

import math
from numbers import Integral


def estimate_bayes_at_n(num_samples, num_correct):
    """Return posterior mean and SD of average problem success probability.

    Each problem has an independent Beta(1, 1) prior. All problems must have
    the same positive number of independently generated programs.

    Args:
        num_samples: Positive integer sample counts, one per problem.
        num_correct: Integer passing-program counts, one per problem.

    Returns:
        A pair of floats, posterior mean and standard deviation on [0, 1].

    Raises:
        ValueError: Counts are empty, unequal, noninteger, or out of range.
    """
    if len(num_samples) == 0 or len(num_samples) != len(num_correct):
        raise ValueError("Bayes@N needs matching, nonempty problem counts")
    for n, c in zip(num_samples, num_correct):
        if (
            isinstance(n, bool)
            or isinstance(c, bool)
            or not isinstance(n, Integral)
            or not isinstance(c, Integral)
            or n <= 0
            or not 0 <= c <= n
        ):
            raise ValueError("Bayes@N needs integer counts with N > 0 and 0 <= c <= N")
    if any(n != num_samples[0] for n in num_samples):
        raise ValueError("Bayes@N needs the same sample count for every problem")
    n = int(num_samples[0])
    total = n + 2
    means = [(int(c) + 1) / total for c in num_correct]
    variances = [
        ((int(c) + 1) / total) * ((n - int(c) + 1) / total) / (total + 1)
        for c in num_correct
    ]
    problems = len(num_samples)
    return math.fsum(means) / problems, math.sqrt(math.fsum(variances)) / problems
