
from functools import lru_cache
from itertools import permutations
from math import factorial
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@lru_cache(maxsize=None)
def _permutation_lookup(m):
    if not 2 <= m <= 7:
        raise ValueError("Supported m: 2 through 7")
    table = np.asarray(list(permutations(range(m))), dtype=np.int64)
    # Base-m encoding of the stable argsort permutation.
    powers = m ** np.arange(m - 1, -1, -1, dtype=np.int64)
    ids = (table * powers).sum(axis=1)
    lookup = np.full(m ** m, -1, dtype=np.int32)
    lookup[ids] = np.arange(factorial(m), dtype=np.int32)
    return lookup, powers, factorial(m)


def rcrspe_vector(raw_voltage, m=6, tau=1, max_scale=20,
                  epsilon=1e-12):
    """Return (entropy_features, ordinal_pattern_coverage), each length B.

    raw_voltage must be the UNNORMALIZED sensor window. Since R/S is scale
    invariant, global affine signal scaling does not change valid results
    except for epsilon-sensitive nearly constant samples.
    """
    x = np.asarray(raw_voltage, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite one-dimensional window")
    if not (2 <= m <= 7 and tau >= 1 and max_scale >= 1 and epsilon > 0):
        raise ValueError("Invalid RCRSPE parameters")
    lookup, powers, n_patterns = _permutation_lookup(m)
    embed_span = (m - 1) * tau + 1
    entropy = np.empty(max_scale, dtype=np.float64)
    coverage = np.empty(max_scale, dtype=np.float64)

    for beta in range(1, max_scale + 1):
        weighted_counts = np.zeros(n_patterns, dtype=np.float64)
        observed_counts = np.zeros(n_patterns, dtype=np.int64)
        for alpha0 in range(beta):  # paper's alpha=1,...,beta
            n_blocks = (x.size - alpha0) // beta
            if n_blocks < embed_span:
                continue
            coarse = x[alpha0:alpha0 + n_blocks * beta].reshape(
                n_blocks, beta
            ).mean(axis=1)
            embeddings = sliding_window_view(
                coarse, embed_span
            )[:, ::tau]
            if embeddings.shape[1] != m:
                raise RuntimeError("Incorrect delay embedding shape")

            permutations0 = np.argsort(
                embeddings, axis=1, kind="stable"
            )
            codes = (permutations0 * powers).sum(axis=1)
            pattern_ids = lookup[codes]
            if np.any(pattern_ids < 0):
                raise RuntimeError("Permutation-index lookup failed")
            amplitude_range = np.ptp(embeddings, axis=1)
            standard_deviation = np.std(embeddings, axis=1, ddof=1)
            weight = amplitude_range / (standard_deviation + epsilon)

            weighted_counts += np.bincount(
                pattern_ids, weights=weight, minlength=n_patterns
            )
            observed_counts += np.bincount(
                pattern_ids, minlength=n_patterns
            )

        denom = float(weighted_counts.sum())
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError(
                f"No positive rescaled-range weights at scale {beta}; "
                "the raw signal may be constant or too short"
            )
        probabilities = weighted_counts[weighted_counts > 0] / denom
        entropy[beta - 1] = -(probabilities * np.log(probabilities)).sum()
        coverage[beta - 1] = np.count_nonzero(observed_counts) / n_patterns

    return entropy, coverage


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t = np.linspace(0, 16*np.pi, 2000)
    signal = np.sin(t) + 0.1*rng.normal(size=2000)
    values, coverage = rcrspe_vector(signal)
    print("RCRSPE:", np.array2string(values, precision=5))
    print("Coverage:", np.array2string(coverage, precision=4))
    assert values.shape == coverage.shape == (20,)
