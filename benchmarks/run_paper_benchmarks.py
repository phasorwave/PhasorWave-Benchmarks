#!/usr/bin/env python3
"""
Reproducible Latency Benchmark Suite for PhasorWave Paper (Table 1).
Batch size fixed at N = 50,000 for all three physical-layer kernels:
  1. ITU-R P.618-14 rain attenuation (N = 50,000 geometries)
  2. ILA-GMP MGS-QR DPD extraction (N = 50,000 samples, P = 39 coefficients)
  3. Optical Turbulence HV-5/7 r_0, theta_0 (N = 50,000 zenith angles)

Measures median latency (ms) across 100 runs after 50 warm-up iterations on:
  (a) Pure Python/NumPy
  (b) Native C++23 AVX2/FMA3 (`phy_engine` via pybind11 zero-copy memoryview)
  (c) WASM SIMD128 (`phy_engine_wasm.js` executed in Node.js)
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "build", Path(__file__).resolve().parent):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import phy_engine

N = 50_000
WARMUP_ITERS = 50
MEASURE_ITERS = 100


def allocate_aligned(num_elements: int, dtype: np.dtype, alignment: int = 64) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    raw = np.empty(num_elements * itemsize + alignment, dtype=np.uint8)
    addr = raw.__array_interface__["data"][0]
    offset = (alignment - (addr % alignment)) % alignment
    return raw[offset : offset + num_elements * itemsize].view(dtype=dtype)


def benchmark_fn(warmup: int, runs: int, fn) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples_ms.append((t1 - t0) / 1e6)
    samples_ms.sort()
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": samples_ms[0],
        "p95_ms": float(np.percentile(samples_ms, 95)),
    }


def numpy_itu_p618_f32(
    r001: np.ndarray,
    h_s: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    f_ghz: float,
    k: float,
    alpha: float,
    p_percent: float,
) -> np.ndarray:
    abs_phi = np.abs(phi_deg)
    h_0 = np.where(abs_phi <= 23.0, 5.0, np.maximum(0.0, 5.0 - 0.075 * (abs_phi - 23.0))).astype(np.float32)
    h_R = h_0 + np.float32(0.36)

    dh = h_R - h_s
    valid_mask = (dh > 0.0) & (r001 > 0.0)
    dh_safe = np.maximum(dh, np.float32(1e-12))
    r001_safe = np.maximum(r001, np.float32(1e-12))

    theta_rad = np.deg2rad(theta_deg).astype(np.float32)
    sin_theta = np.maximum(np.sin(theta_rad), np.float32(1e-12))
    cos_theta = np.maximum(np.cos(theta_rad), np.float32(1e-12))

    L_s_high = dh_safe / sin_theta
    L_s_low = (np.float32(2.0) * dh_safe) / (
        np.sqrt(sin_theta ** 2 + (np.float32(2.0) * dh_safe) / np.float32(8500.0)) + sin_theta
    )
    L_s = np.where(theta_deg >= 5.0, L_s_high, L_s_low).astype(np.float32)
    L_G = L_s * cos_theta
    gamma_R = np.float32(k) * np.power(r001_safe, np.float32(alpha))

    r_001_factor = np.float32(1.0) / (
        np.float32(1.0)
        + np.float32(0.78) * np.sqrt((L_G * gamma_R) / np.float32(f_ghz))
        - np.float32(0.38) * (np.float32(1.0) - np.exp(np.float32(-2.0) * L_G))
    )

    zeta_deg = np.rad2deg(np.arctan(dh_safe / (L_G * r_001_factor))).astype(np.float32)
    L_R = np.where(zeta_deg > theta_deg, (L_G * r_001_factor) / cos_theta, L_s_high)
    chi = np.maximum(np.float32(0.0), np.float32(36.0) - abs_phi)

    v_001_factor = np.float32(1.0) / (
        np.float32(1.0)
        + np.sqrt(sin_theta)
        * (
            np.float32(31.0)
            * (np.float32(1.0) - np.exp(-theta_deg / (np.float32(1.0) + chi)))
            * (np.sqrt(L_R * gamma_R) / np.float32(f_ghz ** 2))
            - np.float32(0.45)
        )
    )

    L_E = L_s * r_001_factor * v_001_factor
    A_001 = gamma_R * L_E

    if p_percent == 0.01:
        A_p = A_001
    else:
        beta = np.zeros_like(A_001)
        if p_percent < 1.0:
            beta_base = np.float32(-0.005) * (abs_phi - np.float32(36.0))
            beta_low = beta_base + np.float32(1.8) - np.float32(4.25) * sin_theta
            beta = np.where(
                abs_phi < 36.0,
                np.where(theta_deg >= 25.0, beta_base, beta_low),
                np.float32(0.0),
            )
        exponent = -(
            np.float32(0.655)
            + np.float32(0.033 * np.log(p_percent))
            - np.float32(0.045) * np.log(np.maximum(A_001, np.float32(1e-12)))
            - beta * np.float32(1.0 - p_percent) * sin_theta
        )
        A_p = A_001 * np.power(np.float32(p_percent / 0.01), exponent)

    return np.where(valid_mask, A_p, np.float32(0.0))


def numpy_hv57_ref(
    zeta_deg: np.ndarray,
    v_wind: float = 21.0,
    a_ground: float = 1.7e-14,
    wavelength: float = 1.55e-6,
    h_max: float = 25000.0,
    num_steps: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(0.0, h_max, num_steps + 1, dtype=np.float64)
    dz = h_max / num_steps
    cn2 = (
        0.00594 * ((v_wind / 27.0) ** 2) * ((1.0e-5 * z) ** 10) * np.exp(-z / 1000.0)
        + 2.7e-16 * np.exp(-z / 1500.0)
        + a_ground * np.exp(-z / 100.0)
    )
    weights = np.ones(num_steps + 1, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= dz / 3.0

    mu_0 = float(np.sum(weights * cn2))
    mu_5_3 = float(np.sum(weights * cn2 * (z ** (5.0 / 3.0))))

    k = (2.0 * np.pi) / wavelength
    k2 = k * k
    sec_zeta = 1.0 / np.cos(np.deg2rad(zeta_deg.astype(np.float64)))
    r_0 = ((0.423 * k2 * sec_zeta * mu_0) ** (-3.0 / 5.0)).astype(np.float32)
    theta_0 = ((2.91 * k2 * (sec_zeta ** (8.0 / 3.0)) * mu_5_3) ** (-3.0 / 5.0)).astype(np.float32)
    return r_0, theta_0


def build_numpy_gmp_matrix_c64(
    u: np.ndarray,
    K: int = 5,
    M: int = 3,
    L: int = 1,
) -> np.ndarray:
    n = u.shape[0]
    env = np.abs(u)
    cols = []

    def shifted_term(sig_delay: int, env_delay: int, env_pow: int) -> np.ndarray:
        col = np.zeros(n, dtype=np.complex64)
        n_start = max(0, sig_delay, env_delay)
        n_end = min(n, n + sig_delay, n + env_delay)
        if n_start < n_end:
            sig_slice = u[n_start - sig_delay : n_end - sig_delay]
            if env_pow == 0:
                col[n_start:n_end] = sig_slice
            else:
                env_slice = env[n_start - env_delay : n_end - env_delay] ** env_pow
                col[n_start:n_end] = sig_slice * env_slice
        return col

    for m_a in range(M):
        for k_a in range(1, K + 1):
            cols.append(shifted_term(m_a, m_a, k_a - 1))

    for m_b in range(M):
        for l_b in range(1, L + 1):
            for k_b in range(2, K + 1):
                cols.append(shifted_term(m_b, m_b + l_b, k_b - 1))

    for m_c in range(M):
        for l_c in range(1, L + 1):
            for k_c in range(2, K + 1):
                cols.append(shifted_term(m_c, m_c - l_c, k_c - 1))

    return np.column_stack(cols)


def numpy_ila_qr_c64(x_desired: np.ndarray, y_pa: np.ndarray, K: int = 5, M: int = 3, L: int = 1) -> np.ndarray:
    U = build_numpy_gmp_matrix_c64(y_pa, K, M, L)
    Q, R = np.linalg.qr(U, mode="reduced")
    d = Q.conj().T @ x_desired
    P = R.shape[1]
    w = np.zeros(P, dtype=np.complex64)
    for i in range(P - 1, -1, -1):
        w[i] = (d[i] - np.dot(R[i, i + 1 :], w[i + 1 :])) / R[i, i]
    return w


def main() -> None:
    if not phy_engine.is_commercial_unlocked():
        lic_path = PROJECT_ROOT / "phasorwave.lic"
        if lic_path.exists():
            phy_engine.verify_license_file(str(lic_path))

    assert phy_engine.is_commercial_unlocked(), f"Commercial module locked: {phy_engine.get_license_status()}"

    print("=" * 88)
    print("  PHASORWAVE PAPER BENCHMARK SUITE (N = 50,000 | 50 WARM-UP + 100 MEASURED RUNS)")
    print("=" * 88)
    print(f"Python / NumPy : {sys.version.split()[0]} / {np.__version__}")
    print(f"OS / Arch      : {platform.platform()} ({platform.machine()})")
    print(f"License status : {phy_engine.get_license_status()}")

    # --------------------------------------------------------------------------
    # 1. ITU-R P.618-14 (N = 50,000)
    # --------------------------------------------------------------------------
    rng_itu = np.random.default_rng(61814)
    r001 = allocate_aligned(N, np.float32)
    h_s = allocate_aligned(N, np.float32)
    theta = allocate_aligned(N, np.float32)
    phi = allocate_aligned(N, np.float32)
    out_itu = allocate_aligned(N, np.float32)

    r001[:] = rng_itu.uniform(5.0, 95.0, size=N).astype(np.float32)
    h_s[:] = rng_itu.uniform(0.0, 3.2, size=N).astype(np.float32)
    theta[:] = rng_itu.uniform(5.0, 85.0, size=N).astype(np.float32)
    phi[:] = rng_itu.uniform(-60.0, 60.0, size=N).astype(np.float32)

    mv_r = memoryview(r001)
    mv_hs = memoryview(h_s)
    mv_th = memoryview(theta)
    mv_ph = memoryview(phi)
    mv_itu_out = memoryview(out_itu)

    itu_model = phy_engine.ITU_P618_RainModel(f=20.0, k=0.0951, alpha=1.0080, p=0.05)

    itu_py = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: numpy_itu_p618_f32(r001, h_s, theta, phi, 20.0, 0.0951, 1.0080, 0.05),
    )
    itu_cpp = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: itu_model.compute_attenuation(mv_r, mv_hs, mv_th, mv_ph, mv_itu_out),
    )

    # --------------------------------------------------------------------------
    # 2. ILA-GMP MGS-QR (N = 50,000, P = 39)
    # --------------------------------------------------------------------------
    dpd_model = phy_engine.DPD_GMP_Model(K=5, M=3, L=1)
    P = dpd_model.num_coefficients
    assert P == 39

    rng_dpd = np.random.default_rng(2026)
    x_des = allocate_aligned(N, np.complex64)
    y_pa = allocate_aligned(N, np.complex64)
    w_out = allocate_aligned(P, np.complex64)
    lin_out = allocate_aligned(N, np.complex64)

    raw_iq = rng_dpd.normal(0.0, 0.32, size=N) + 1j * rng_dpd.normal(0.0, 0.32, size=N)
    mag = np.abs(raw_iq)
    x_des[:] = np.where(mag > 0.85, raw_iq * (0.85 / mag), raw_iq).astype(np.complex64)
    env = np.abs(x_des)
    y_pa[:] = (
        x_des
        + (-0.18 + 0.07j) * x_des * (env ** 2)
        + (-0.06 - 0.03j) * x_des * (env ** 4)
    ).astype(np.complex64)

    mv_x = memoryview(x_des)
    mv_y = memoryview(y_pa)
    mv_w = memoryview(w_out)
    mv_lin = memoryview(lin_out)

    dpd_py = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: numpy_ila_qr_c64(x_des, y_pa, K=5, M=3, L=1),
    )
    dpd_cpp = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: dpd_model.extract_coefficients(mv_x, mv_y, mv_w),
    )
    dpd_cpp_apply = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: (dpd_model.extract_coefficients(mv_x, mv_y, mv_w), dpd_model.apply_dpd(mv_y, mv_w, mv_lin)),
    )

    # --------------------------------------------------------------------------
    # 3. HV-5/7 Optical Turbulence (N = 50,000)
    # --------------------------------------------------------------------------
    rng_opt = np.random.default_rng(1550)
    zeta = allocate_aligned(N, np.float32)
    out_r0 = allocate_aligned(N, np.float32)
    out_th0 = allocate_aligned(N, np.float32)
    zeta[:] = rng_opt.uniform(0.0, 75.0, size=N).astype(np.float32)

    mv_zeta = memoryview(zeta)
    mv_r0 = memoryview(out_r0)
    mv_th0 = memoryview(out_th0)

    hv_model = phy_engine.OpticalTurbulence_HV57(
        v=21.0, A=1.7e-14, wavelength=1.55e-6, h_max=25000.0, num_steps=4000, zeta_degrees=True
    )

    hv_py = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: numpy_hv57_ref(zeta, 21.0, 1.7e-14, 1.55e-6, 25000.0, 4000),
    )
    hv_cpp = benchmark_fn(
        WARMUP_ITERS,
        MEASURE_ITERS,
        lambda: hv_model.compute_parameters(mv_zeta, mv_r0, mv_th0),
    )

    # --------------------------------------------------------------------------
    # 4. WASM SIMD128 via Node.js
    # --------------------------------------------------------------------------
    wasm_runner = Path(__file__).resolve().parent / "run_wasm_benchmarks.cjs"
    proc = subprocess.run(
        ["node", str(wasm_runner)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    wasm_data = json.loads(proc.stdout)

    itu_wasm = wasm_data["kernels"]["itu_p618"]
    dpd_wasm = wasm_data["kernels"]["dpd_ila_qr"]
    hv_wasm = wasm_data["kernels"]["hv57_optical"]

    wasm_size_bytes = wasm_data.get(
        "wasm_size_bytes",
        (PROJECT_ROOT / "build" / "wasm-benchmark" / "phy_engine_wasm.wasm").stat().st_size,
    )
    print("-" * 88)
    print(f"Node Runtime   : {wasm_data['runtime']}")
    print(f"WASM Binary    : {wasm_size_bytes:,} bytes")
    print("-" * 88)
    print(f"{'Kernel (N = 50,000)':<38} | {'Python/NumPy':>12} | {'C++ AVX2/FMA3':>13} | {'WASM SIMD128':>13}")
    print("-" * 88)
    print(
        f"{'ITU-R P.618-14 (geometries)':<38} | "
        f"{itu_py['median_ms']:9.2f} ms | {itu_cpp['median_ms']:10.2f} ms | {itu_wasm['median_ms']:10.2f} ms"
    )
    print(
        f"{'ILA-GMP MGS-QR (P = 39)':<38} | "
        f"{dpd_py['median_ms']:9.2f} ms | {dpd_cpp['median_ms']:10.2f} ms | {dpd_wasm['median_ms']:10.2f} ms"
    )
    print(
        f"{'HV-5/7 r0, theta0 (angles)':<38} | "
        f"{hv_py['median_ms']:9.2f} ms | {hv_cpp['median_ms']:10.2f} ms | {hv_wasm['median_ms']:10.2f} ms"
    )
    print("-" * 88)
    print(f"Note: C++ extract+apply median = {dpd_cpp_apply['median_ms']:.2f} ms vs extract-only = {dpd_cpp['median_ms']:.2f} ms")
    print(f"Speedups (Py vs C++ AVX2): ITU={itu_py['median_ms']/itu_cpp['median_ms']:.1f}x, "
          f"DPD={dpd_py['median_ms']/dpd_cpp['median_ms']:.1f}x, "
          f"HV57={hv_py['median_ms']/hv_cpp['median_ms']:.1f}x")
    print(f"Speedups (Py vs WASM SIMD128): ITU={itu_py['median_ms']/itu_wasm['median_ms']:.1f}x, "
          f"DPD={dpd_py['median_ms']/dpd_wasm['median_ms']:.1f}x, "
          f"HV57={hv_py['median_ms']/hv_wasm['median_ms']:.1f}x")

    summary = {
        "N": N,
        "warmup": WARMUP_ITERS,
        "runs": MEASURE_ITERS,
        "itu_p618": {"numpy": itu_py, "cpp_avx2": itu_cpp, "wasm_simd128": itu_wasm},
        "dpd_ila_qr": {
            "numpy": dpd_py,
            "cpp_avx2_extract": dpd_cpp,
            "cpp_avx2_extract_and_apply": dpd_cpp_apply,
            "wasm_simd128_extract_and_apply": dpd_wasm,
        },
        "hv57_optical": {"numpy": hv_py, "cpp_avx2": hv_cpp, "wasm_simd128": hv_wasm},
    }
    print("\nJSON_SUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
