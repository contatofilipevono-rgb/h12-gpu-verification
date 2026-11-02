"""Busca exaustiva de RS bent homogeneas em n variaveis (GPU, cupy RawKernel).

Uso no Colab:  run(n, degrees)  ->  dict com contagens e solucoes verificadas.
  - percorre TODAS as combinacoes de orbitas ANF dos graus pedidos (codigo Gray);
  - filtro na GPU: |W_f(u)| = 2^(n/2) em u = 0 e em alguns u de controle;
  - sobreviventes sao verificados na CPU com espectro de Walsh completo.
Controles: run(12,[2]) deve achar 16 bent; run(12,[3]) e run(8,[4]) devem achar 0.
"""
import itertools, time, json
import numpy as np
import cupy as cp

def orbits(n, d):
    seen, out = set(), []
    for S in itertools.combinations(range(n), d):
        if S in seen:
            continue
        orb = {tuple(sorted((s + t) % n for s in S)) for t in range(n)}
        seen |= orb
        out.append(sorted(orb))
    return out

def table_words(n, monos):
    x = np.arange(1 << n, dtype=np.int64)
    f = np.zeros(1 << n, dtype=np.uint8)
    for m in monos:
        mask = sum(1 << i for i in m)
        f ^= ((x & mask) == mask).astype(np.uint8)
    return np.packbits(f.reshape(-1, 64)[:, ::-1], axis=1).view(">u8").astype(np.uint64).ravel(), f

def linear_words(n, u):
    x = np.arange(1 << n, dtype=np.int64)
    f = np.array([bin(v & u).count("1") & 1 for v in x], dtype=np.uint8)
    return np.packbits(f.reshape(-1, 64)[:, ::-1], axis=1).view(">u8").astype(np.uint64).ravel()

KERNEL = r"""
extern "C" __global__
void search(const unsigned long long* __restrict__ basis, int K, int LOW,
            const unsigned long long* __restrict__ checks, int NCHK,
            int wlo, int whi, long long prefix0, long long nthreads,
            unsigned long long* out, unsigned long long* nout, int maxout,
            unsigned long long* npass0)
{
    extern __shared__ unsigned long long sb[];
    for (int i = threadIdx.x; i < K * WORDS; i += blockDim.x) sb[i] = basis[i];
    for (int i = threadIdx.x; i < NCHK * WORDS; i += blockDim.x) sb[K * WORDS + i] = checks[i];
    __syncthreads();
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;
    long long prefix = prefix0 + tid;
    unsigned long long T[WORDS];
    #pragma unroll
    for (int w = 0; w < WORDS; w++) T[w] = 0ULL;
    for (int j = 0; j < K - LOW; j++)
        if ((prefix >> j) & 1)
            #pragma unroll
            for (int w = 0; w < WORDS; w++) T[w] ^= sb[(LOW + j) * WORDS + w];
    unsigned long long local_pass = 0;
    long long steps = 1LL << LOW;
    for (long long s = 0; s < steps; s++) {
        int wt = 0;
        #pragma unroll
        for (int w = 0; w < WORDS; w++) wt += __popcll(T[w]);
        if (wt == wlo || wt == whi) {
            local_pass++;
            bool ok = true;
            for (int c = 0; c < NCHK && ok; c++) {
                int w2 = 0;
                #pragma unroll
                for (int w = 0; w < WORDS; w++) w2 += __popcll(T[w] ^ sb[(K + c) * WORDS + w]);
                ok = (w2 == wlo || w2 == whi);
            }
            if (ok) {
                unsigned long long k = atomicAdd(nout, 1ULL);
                if (k < (unsigned long long)maxout)
                    out[k] = ((unsigned long long)prefix << LOW) | (unsigned long long)(s ^ (s >> 1));
            }
        }
        if (s + 1 < steps) {
            int b = __ffsll(s + 1) - 1;
            #pragma unroll
            for (int w = 0; w < WORDS; w++) T[w] ^= sb[b * WORDS + w];
        }
    }
    atomicAdd(npass0, local_pass);
}
"""

def fwht(a):
    a = a.astype(np.int64).copy(); h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack((a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]), axis=1).reshape(-1)
        h *= 2
    return a

def run(n, degrees, low=24, batch_threads=1 << 16, maxout=1 << 20, verbose=True):
    t0 = time.time()
    orbs = [o for d in degrees for o in orbits(n, d)]
    K = len(orbs)
    LOW = min(low, K)
    words = (1 << n) // 64
    tabs, f8 = zip(*[table_words(n, o) for o in orbs])
    basis = cp.asarray(np.concatenate(tabs))
    us = [(1 << n) - 1, 1, 3, 5, 0b1001, (1 << (n // 2)) | 1, 0b10101]
    us = [u for u in us if u < (1 << n)]
    checks = cp.asarray(np.concatenate([linear_words(n, u) for u in us]))
    half = 1 << (n - 1); dev = 1 << (n // 2 - 1)
    mod = cp.RawModule(code=KERNEL, options=(f"-DWORDS={words}",))
    ker = mod.get_function("search")
    out = cp.zeros(maxout, dtype=cp.uint64)
    nout = cp.zeros(1, dtype=cp.uint64); npass = cp.zeros(1, dtype=cp.uint64)
    total_threads = 1 << (K - LOW)
    shmem = (K + len(us)) * words * 8
    done = 0
    while done < total_threads:
        nt = min(batch_threads, total_threads - done)
        block = 128
        grid = (nt + block - 1) // block
        ker((grid,), (block,), (basis, np.int32(K), np.int32(LOW), checks, np.int32(len(us)),
                                np.int32(half - dev), np.int32(half + dev), np.int64(done), np.int64(nt),
                                out, nout, np.int32(maxout), npass), shared_mem=shmem)
        cp.cuda.Stream.null.synchronize()
        done += nt
        if verbose:
            print(f"n={n} deg={degrees}: {done}/{total_threads} blocos-thread, "
                  f"sobreviventes={int(nout[0])}, passa W(0)={int(npass[0])}, {time.time()-t0:.0f}s", flush=True)
    ns = int(nout[0])
    cand = cp.asnumpy(out[:min(ns, maxout)])
    bent = []
    F = [f.astype(np.uint8) for f in f8]
    for c in cand:
        f = np.zeros(1 << n, dtype=np.uint8)
        for j in range(K):
            if (int(c) >> j) & 1:
                f ^= F[j]
        W = fwht(1 - 2 * f.astype(np.int64))
        if np.all(np.abs(W) == 1 << (n // 2)):
            bent.append([orbs[j][0] for j in range(K) if (int(c) >> j) & 1])
    res = {"n": n, "degrees": degrees, "anf_orbits": K, "functions": 2 ** K,
           "pass_weight_filter": int(npass[0]), "gpu_survivors": ns,
           "overflow": ns > maxout, "bent_verified": len(bent), "bent_examples": bent[:20],
           "check_u": us, "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(json.dumps(res))
    return res
