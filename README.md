# PhasorWave Benchmarks (`phasorwave-benchmarks`)

Official numerical validation and latency benchmark suite accompanying the paper:

> **"Accelerating Indirect Learning Architecture (ILA) Digital Predistortion and LEO Optical Link Simulations via 128-Bit Vector SIMD Execution in Hybrid WebAssembly and C++23 Environments"**

This repository provides the transparent physical-layer reference models and reproducible benchmarking harness used to generate **Table 1** (end-to-end latency across Python/NumPy, native C++23 AVX2/FMA3, and WASM SIMD128) and **Table 2** (double-precision numerical fidelity, GMP behavioral modeling NMSE, and closed-loop ILA ACPR linearization).

---

## Interactive WebAssembly Simulator

Test the **WASM SIMD128** physical-layer simulator directly in your browser with zero installation at:

**https://phasorwave.dev**

The public web platform runs WebAssembly SIMD128 kernels locally in client memory without cloud round-trips.

---

## Repository Structure & Requirements

### Repository Layout

```text
phasorwave-benchmarks/
├── benchmarks/
│   ├── run_paper_benchmarks.py      # Table 1 latency suite (N = 50,000; Python/NumPy, C++23 AVX2, WASM SIMD128)
│   ├── run_accuracy_validation.py   # Table 2 numerical & spectral validation (ITU-R P.618-14, HV-5/7, ILA-GMP DPD)
│   └── run_wasm_benchmarks.cjs      # Node.js WASM SIMD128 benchmark runner (50 warm-up + 100 measured iterations)
├── LICENSE
└── README.md
```

### Script Descriptions

- **`benchmarks/run_paper_benchmarks.py`**: Benchmarks median, mean, minimum, and P95 latency (`ms`) over **100 measured runs** after **50 warm-up iterations** at fixed batch size `N = 50,000` for:
  1. **ITU-R P.618-14** slant-path rain attenuation (`50,000` link geometries).
  2. **ILA-GMP MGS-QR DPD** extraction (`K = 5, M = 3, L = 1`, `P = 39` complex coefficients over `50,000` samples).
  3. **Hufnagel–Valley (HV-5/7)** optical turbulence moments & coherence parameters (`r_0`, `theta_0` over `50,000` zenith angles and `4,000` Simpson steps).
  Contains complete self-contained vectorized Python/NumPy baselines (`numpy_itu_p618_f32`, `numpy_ila_qr_c64`, `numpy_hv57_ref`) alongside native `phy_engine` execution and automated invocation of `run_wasm_benchmarks.cjs`.

- **`benchmarks/run_accuracy_validation.py`**: Evaluates single-precision (`float32`) SIMD accuracy and RF PA linearization over `N = 50,000`:
  1. Max and mean absolute error (dB) of ITU-R P.618-14 $A_p$ against an analytical `float64` NumPy reference (`python_f64_itu_p618`).
  2. Max relative error of HV-5/7 $r_0$ and $\theta_0$ against a 4,000-step Simpson `float64` NumPy reference (`python_f64_hv57`).
  3. Forward GMP model identification NMSE (dB) against a Wiener–Hammerstein nonlinear PA (`simulate_physical_pa`, Rapp AM/AM + Saleh-like AM/PM + memory/cross-memory) driven by an 8x-oversampled RRC-filtered 64-QAM signal (`beta = 0.22`).
  4. Worst-case and average adjacent channel power ratio (ACPR, dBc / dB improvement) computed via Hann-windowed Welch PSD (`nfft = 4096`, 50% overlap) over `n = 3` closed-loop ILA iterations.

- **`benchmarks/run_wasm_benchmarks.cjs`**: Standalone CommonJS benchmark runner executed under Node.js (V8 WebAssembly SIMD128) using built-in `node:assert/strict`, `node:fs`, and `node:path` modules only. Aborts immediately if a restricted Community WASM build returns stub error codes (`-1` or `-2`) or `dpd_num_coeffs !== 39`.

### Requirements

#### Python/NumPy Baselines & Reference Models
- **Python** `>= 3.10`
- **NumPy** `>= 1.26` (`numpy`) — all reference baselines use pure NumPy and the Python standard library (no SciPy dependency).

```bash
pip install "numpy>=1.26"
```

#### Full Native & WASM End-to-End Execution
- **PhasorWave Native SDK (`phy_engine`)**: Required by `run_paper_benchmarks.py` and `run_accuracy_validation.py` to execute the C++23 AVX2/FMA3 kernels (`ITU_P618_RainModel`, `OpticalTurbulence_HV57`, `DPD_GMP_Model`) and an activated Enterprise offline license (`phasorwave.lic`). Hardware must support **x86-64 AVX2 and FMA3**.
- **Node.js** (`>= 20`, tested on `v24.x`): Required by `run_wasm_benchmarks.cjs` (and invoked automatically by `run_paper_benchmarks.py`) along with a full-kernel benchmark build of `phy_engine_wasm.js` / `phy_engine_wasm.wasm`.

---

## Reproducing the Benchmarks

### 1. Full Latency Suite (Table 1)

From the root of the workspace (with the `phy_engine` module, active `phasorwave.lic`, Node.js, and the benchmark `.wasm` binary available):

```bash
python benchmarks/run_paper_benchmarks.py
```

Neither Python benchmark script takes CLI flags; both read fixed reproducible RNG seeds (`61814`, `2026`, `1550`) and batch size `N = 50,000` (`50` warm-up, `100` measured iterations). Output ends with a structured `JSON_SUMMARY:` block.

### 2. Standalone WASM SIMD128 Runner (Node.js)

By default, `run_wasm_benchmarks.cjs` looks for `../build/wasm-benchmark/phy_engine_wasm.js`. You can pass a custom benchmark WASM bundle path via `--wasm`, positional `.js` argument, or the `PHASORWAVE_WASM_JS` environment variable:

```bash
# Default lookup (build/wasm-benchmark/phy_engine_wasm.js)
node benchmarks/run_wasm_benchmarks.cjs

# Explicit path to the benchmark WASM factory
node benchmarks/run_wasm_benchmarks.cjs --wasm /path/to/phy_engine_wasm.js

# Or via environment variable
PHASORWAVE_WASM_JS=/path/to/phy_engine_wasm.js node benchmarks/run_wasm_benchmarks.cjs
```

Outputs machine-readable JSON metrics (`wasm_size_bytes`, `dpd_num_coeffs`, `dpd_ila_flops`, and `median_ms`/`mean_ms`/`min_ms`/`p95_ms` for `itu_p618`, `dpd_ila_qr`, and `hv57_optical`).

### 3. Numerical Accuracy & Spectral Linearization Suite (Table 2)

Validates `ITU-R P.618-14`, `HV-5/7`, and 3-iteration closed-loop `ILA-GMP` DPD accuracy (printing a structured `JSON_ACCURACY:` summary):

```bash
python benchmarks/run_accuracy_validation.py
```

---

## Commercial Native C++23 SDK

While this public repository exposes the benchmark methodology, WASM harness, and Python/NumPy baselines, the precompiled native **C++23 AVX2/FMA3 SDK** (`phy_engine`)—featuring zero-copy PEP 3118 `memoryview` bindings, `float64` (`__m256d`) double-precision paths, and GNU Radio 4.0 blocks—is distributed commercially with hardware-bound offline **Ed25519** licensing (activated once online via `python -m phy_engine activate <LICENSE_KEY>` and executed 100% offline thereafter).

Licensing plans and trial inquiries are available at **https://phasorwave.dev**. See [LICENSE](LICENSE) for full Open-Core (AGPLv3 / MIT) and commercial licensing terms.

---

## Citation (BibTeX)

```bibtex
@misc{phasorwave2026simd,
  author       = {{PhasorWave Engineering Team}},
  title        = {Accelerating Indirect Learning Architecture ({ILA}) Digital Predistortion and {LEO} Optical Link Simulations via 128-Bit Vector {SIMD} Execution in Hybrid {WebAssembly} and {C++23} Environments},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://phasorwave.dev}},
  note         = {Reproducibility benchmark suite: \texttt{phasorwave-benchmarks}}
}
```