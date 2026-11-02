# Independent GPU verification: no homogeneous quartic RS bent function in 12 variables

This is an exhaustive search over all 2^43 homogeneous degree-4 rotation symmetric (RS) Boolean functions in 12 variables. **None is bent.**

The search independently confirms the result of F. Vono, *Nonexistence of homogeneous rotation symmetric bent functions of degree four in twelve variables* (figshare 34737294, 2026). The code is written from scratch and shares nothing with that paper's proof.

## Result (`results/h12_quartic_independente_a100.json`)

| Item | Value |
|---|---:|
| Quartic monomial orbits under rotation | 43 |
| Functions enumerated | 2^43 = 8,796,093,022,208 |
| Pass the weight filter (\|W(0)\| = 64) | 138,735,411,138 |
| Pass 7 more Walsh checks on GPU | 34,748 |
| **Bent (full Walsh spectrum on CPU)** | **0** |
| Wall time, Colab A100 | 1788 s |

## Method (`gpu_homog_rs_bent.py`)

1. **Truth tables.** Each orbit's truth table is held as 4096 bits (64 words).
2. **Enumeration.** A CUDA kernel (cupy `RawKernel`) walks all subsets of orbits in Gray-code order. Each step is one XOR per word.
3. **GPU filter.** A function survives if |W(u)| = 64 for u = 0, the all-ones vector, and the check points 1, 3, 5, 9, 65 and 21, each computed by popcount.
4. **CPU verification.** Every survivor is tested with the full fast Walsh–Hadamard transform.

## Controls (`results/gpu_controles_a100.json`)

| n | degree | functions | bent found | expected |
|---|---|---:|---:|---|
| 12 | 2 | 64 | 16 | 16, matches an independent CPU count |
| 12 | 3 | 2^19 | 0 | 0, cubic case known |
| 8 | 4 | 2^10 | 0 | 0, degree n/2 (Xia et al. 2004) |
| 8 | 2 | 16 | 8 | 8 |

## Reproduce

```python
# Colab with a GPU runtime and cupy installed
from gpu_homog_rs_bent import run
run(12, [2])   # control: 16 bent
run(12, [4])   # main search: about 30 min on an A100
```

Authors: Filipe Vono, with computations by Claude (Anthropic), 2 November 2026.
