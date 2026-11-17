#!/usr/bin/env python3
"""
Numerical Accuracy and Spectral Linearization Validation for PhasorWave Paper (Table 2).
Computes:
  1. Max. absolute error in A_p (ITU-R P.618-14, float32 SIMD vs. Python float64) over N = 50,000 geometries.
  2. Max. relative error in r_0 (HV-5/7, float32 SIMD vs. Python float64) over N = 50,000 zenith angles.
  3. NMSE (dB) of the QR-identified GMP behavioral model (K=5, M=3, L=1, P=39) against nonlinear PA output.
  4. ACPR improvement (dB) via Welch PSD after n = 3 ILA iterations on a wideband 8x-oversampled 64-QAM signal.
"""

from __future__ import annotations

import json
import math
import sys
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


def allocate_aligned(num_elements: int, dtype: np.dtype, alignment: int = 64) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    raw = np.empty(num_elements * itemsize + alignment, dtype=np.uint8)
    addr = raw.__array_interface__["data"][0]
    offset = (alignment - (addr % alignment)) % alignment
    return raw[offset : offset + num_elements * itemsize].view(dtype=dtype)


def format_latex_sci(val: float, unit: str = "") -> str:
    if val == 0.0:
        return f"$0.00{unit}$"
    exp = math.floor(math.log10(abs(val)))
    mant = val / (10.0 ** exp)
    if mant >= 9.995:
        mant /= 10.0
        exp += 1
    unit_str = rf"\text{{ {unit}}}" if unit else ""
    return rf"${mant:.2f} \times 10^{{{exp}}}{unit_str}$"


def python_f64_itu_p618(
    r001: np.ndarray,
    h_s: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    f_ghz: float,
    k: float,
    alpha: float,
    p_percent: float,
) -> np.ndarray:
    r001 = r001.astype(np.float64)
    h_s = h_s.astype(np.float64)
    theta_deg = theta_deg.astype(np.float64)
    phi_deg = phi_deg.astype(np.float64)

    abs_phi = np.abs(phi_deg)
    h_0 = np.where(abs_phi <= 23.0, 5.0, np.maximum(0.0, 5.0 - 0.075 * (abs_phi - 23.0)))
    h_R = h_0 + 0.36

    dh = h_R - h_s
    valid = (dh > 0.0) & (r001 > 0.0)
    dh_safe = np.maximum(dh, 1e-12)
    r001_safe = np.maximum(r001, 1e-12)

    theta_rad = np.deg2rad(theta_deg)
    sin_theta = np.maximum(np.sin(theta_rad), 1e-12)
    cos_theta = np.maximum(np.cos(theta_rad), 1e-12)

    L_s_high = dh_safe / sin_theta
    L_s_low = (2.0 * dh_safe) / (np.sqrt(sin_theta ** 2 + (2.0 * dh_safe) / 8500.0) + sin_theta)
    L_s = np.where(theta_deg >= 5.0, L_s_high, L_s_low)
    L_G = L_s * cos_theta
    gamma_R = k * np.power(r001_safe, alpha)

    r_001_factor = 1.0 / (
        1.0 + 0.78 * np.sqrt((L_G * gamma_R) / f_ghz) - 0.38 * (1.0 - np.exp(-2.0 * L_G))
    )

    zeta_deg = np.rad2deg(np.arctan(dh_safe / (L_G * r_001_factor)))
    L_R = np.where(zeta_deg > theta_deg, (L_G * r_001_factor) / cos_theta, L_s_high)
    chi = np.maximum(0.0, 36.0 - abs_phi)

    v_001_factor = 1.0 / (
        1.0
        + np.sqrt(sin_theta)
        * (31.0 * (1.0 - np.exp(-theta_deg / (1.0 + chi))) * (np.sqrt(L_R * gamma_R) / (f_ghz ** 2)) - 0.45)
    )

    L_E = L_s * r_001_factor * v_001_factor
    A_001 = gamma_R * L_E

    if p_percent == 0.01:
        A_p = A_001
    else:
        beta = np.zeros_like(A_001)
        if p_percent < 1.0:
            beta_base = -0.005 * (abs_phi - 36.0)
            beta_low = beta_base + 1.8 - 4.25 * sin_theta
            beta = np.where(abs_phi < 36.0, np.where(theta_deg >= 25.0, beta_base, beta_low), 0.0)
        exponent = -(
            0.655
            + 0.033 * np.log(p_percent)
            - 0.045 * np.log(np.maximum(A_001, 1e-12))
            - beta * (1.0 - p_percent) * sin_theta
        )
        A_p = A_001 * np.power(p_percent / 0.01, exponent)

    return np.where(valid, A_p, 0.0)


def python_f64_hv57(
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
    r_0 = (0.423 * k2 * sec_zeta * mu_0) ** (-3.0 / 5.0)
    theta_0 = (2.91 * k2 * (sec_zeta ** (8.0 / 3.0)) * mu_5_3) ** (-3.0 / 5.0)
    return r_0, theta_0


def generate_rrc_64qam_signal(num_samples: int, sps: int = 8, beta: float = 0.22, seed: int = 2026) -> np.ndarray:
    """Generates a stationary 8x-oversampled 64-QAM baseband signal via circular RRC pulse shaping."""
    assert num_samples % sps == 0
    num_symbols = num_samples // sps
    rng = np.random.default_rng(seed)

    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
    sym_i = rng.choice(levels, size=num_symbols)
    sym_q = rng.choice(levels, size=num_symbols)
    syms = (sym_i + 1j * sym_q) / math.sqrt(42.0)

    upsampled = np.zeros(num_samples, dtype=np.complex128)
    upsampled[::sps] = syms

    # Frequency-domain exact Root-Raised-Cosine response (periodic, zero edge transient)
    freqs = np.fft.fftfreq(num_samples, d=1.0 / sps)  # normalized to symbol rate Rs = 1.0
    abs_f = np.abs(freqs)
    H = np.zeros(num_samples, dtype=np.float64)
    f_pass = (1.0 - beta) / 2.0
    f_stop = (1.0 + beta) / 2.0
    H[abs_f <= f_pass] = 1.0
    trans = (abs_f > f_pass) & (abs_f <= f_stop)
    H[trans] = np.sqrt(0.5 * (1.0 + np.cos((np.pi / beta) * (abs_f[trans] - f_pass))))

    sig = np.fft.ifft(np.fft.fft(upsampled) * H) * math.sqrt(sps)
    # Normalize RMS to 0.36 (~7.5 dB PAPR -> peak ~0.85) to drive the PA near compression
    sig *= 0.36 / np.sqrt(np.mean(np.abs(sig) ** 2))
    return sig.astype(np.complex64)


def simulate_physical_pa(x: np.ndarray) -> np.ndarray:
    """
    Physical nonlinear RF power amplifier with Wiener-Hammerstein memory and cross-memory:
      1. Input thermal/bias FIR memory filter
      2. Smooth Rapp AM/AM compression (A_sat = 0.88, p = 1.2) + Saleh-like AM/PM phase distortion
      3. Post-nonlinear memory tap and cross-envelope memory coupling
    """
    x64 = x.astype(np.complex128)

    # 1. Pre-memory (causal FIR)
    v = x64.copy()
    v[1:] += (0.060 - 0.025j) * x64[:-1]
    v[2:] += (0.018 + 0.009j) * x64[:-2]

    # 2. Non-polynomial Rapp AM/AM + AM/PM
    r = np.abs(v)
    a_sat = 0.88
    p_rapp = 1.2
    am_am = 1.0 / np.power(1.0 + np.power(r / a_sat, 2.0 * p_rapp), 1.0 / (2.0 * p_rapp))
    phi_pm = (0.38 * (r ** 2)) / (1.0 + 0.40 * (r ** 2))
    w = v * am_am * np.exp(1j * phi_pm)

    # 3. Post-memory + lagging cross-envelope term
    y = w.copy()
    y[1:] += (0.030 - 0.015j) * w[:-1]
    env_x = np.abs(x64)
    y[1:] += (-0.035 + 0.018j) * x64[1:] * (env_x[:-1] ** 2)
    return y.astype(np.complex64)


def welch_psd(signal: np.ndarray, nfft: int = 4096, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy Welch PSD estimator with Hann window (two-sided, fftshifted)."""
    sig = signal.astype(np.complex128)
    step = int(nfft * (1.0 - overlap))
    win = np.hanning(nfft)
    win_norm = np.sum(win ** 2)

    acc = np.zeros(nfft, dtype=np.float64)
    count = 0
    for start in range(0, len(sig) - nfft + 1, step):
        seg = sig[start : start + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg, n=nfft))
        acc += (np.abs(spec) ** 2) / win_norm
        count += 1

    psd = acc / max(count, 1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))  # normalized to [-0.5, 0.5)
    return freqs, psd


def compute_acpr_dbc(signal: np.ndarray, sps: int = 8) -> tuple[float, float, float]:
    """Computes lower, upper, and worst-case ACPR (dBc) via Welch PSD."""
    freqs, psd = welch_psd(signal, nfft=4096, overlap=0.5)
    # Symbol rate Rs = 1/sps = 0.125; RRC occupied BW = 1.22*Rs = 0.1525 (+/- 0.07625)
    main_mask = np.abs(freqs) <= 0.080
    adj_lo_mask = (freqs >= -0.240) & (freqs <= -0.090)
    adj_hi_mask = (freqs >= 0.090) & (freqs <= 0.240)

    p_main = float(np.sum(psd[main_mask]))
    p_lo = float(np.sum(psd[adj_lo_mask]))
    p_hi = float(np.sum(psd[adj_hi_mask]))

    acpr_lo = 10.0 * math.log10(max(p_lo / p_main, 1e-18))
    acpr_hi = 10.0 * math.log10(max(p_hi / p_main, 1e-18))
    return acpr_lo, acpr_hi, max(acpr_lo, acpr_hi)


def compute_nmse_db(ref: np.ndarray, est: np.ndarray) -> float:
    err = float(np.sum(np.abs(ref.astype(np.complex128) - est.astype(np.complex128)) ** 2))
    pwr = float(np.sum(np.abs(ref.astype(np.complex128)) ** 2))
    return 10.0 * math.log10(max(err / max(pwr, 1e-18), 1e-18))


def main() -> None:
    if not phy_engine.is_commercial_unlocked():
        lic_path = PROJECT_ROOT / "phasorwave.lic"
        if lic_path.exists():
            phy_engine.verify_license_file(str(lic_path))
    assert phy_engine.is_commercial_unlocked()

    print("=" * 88)
    print("  PHASORWAVE ACCURACY & SPECTRAL LINEARIZATION VALIDATION (TABLE 2)")
    print("=" * 88)

    # --------------------------------------------------------------------------
    # 1. ITU-R P.618-14: Max absolute error in A_p (f32 SIMD vs f64)
    # --------------------------------------------------------------------------
    rng_itu = np.random.default_rng(61814)
    r001 = allocate_aligned(N, np.float32)
    h_s = allocate_aligned(N, np.float32)
    theta = allocate_aligned(N, np.float32)
    phi = allocate_aligned(N, np.float32)
    ap_f32 = allocate_aligned(N, np.float32)

    r001[:] = rng_itu.uniform(5.0, 95.0, size=N).astype(np.float32)
    h_s[:] = rng_itu.uniform(0.0, 3.2, size=N).astype(np.float32)
    theta[:] = rng_itu.uniform(5.0, 85.0, size=N).astype(np.float32)
    phi[:] = rng_itu.uniform(-60.0, 60.0, size=N).astype(np.float32)
    h_s[100:116] = 6.5
    r001[200:216] = 0.0

    itu_model = phy_engine.ITU_P618_RainModel(f=20.0, k=0.0951, alpha=1.0080, p=0.05)
    itu_model.compute_attenuation(
        memoryview(r001), memoryview(h_s), memoryview(theta), memoryview(phi), memoryview(ap_f32)
    )
    ap_f64 = python_f64_itu_p618(r001, h_s, theta, phi, 20.0, 0.0951, 1.0080, 0.05)

    itu_abs_diff = np.abs(ap_f32.astype(np.float64) - ap_f64)
    itu_max_abs_err = float(np.max(itu_abs_diff))
    itu_mean_abs_err = float(np.mean(itu_abs_diff))

    print(f"[1] ITU-R P.618-14 (N = {N:,}, f32 SIMD vs f64):")
    print(f"    Max. absolute error (A_p)  : {itu_max_abs_err:.6e} dB  -> {format_latex_sci(itu_max_abs_err, 'dB')}")
    print(f"    Mean absolute error (A_p)  : {itu_mean_abs_err:.6e} dB")

    # --------------------------------------------------------------------------
    # 2. HV-5/7 Optical Turbulence: Max relative error in r_0 (f32 SIMD vs f64)
    # --------------------------------------------------------------------------
    rng_opt = np.random.default_rng(1550)
    zeta = allocate_aligned(N, np.float32)
    r0_f32 = allocate_aligned(N, np.float32)
    th0_f32 = allocate_aligned(N, np.float32)

    zeta[:] = rng_opt.uniform(0.0, 75.0, size=N).astype(np.float32)
    zeta[:5] = np.array([0.0, 15.0, 30.0, 45.0, 60.0], dtype=np.float32)

    hv_model = phy_engine.OpticalTurbulence_HV57(
        v=21.0, A=1.7e-14, wavelength=1.55e-6, h_max=25000.0, num_steps=4000, zeta_degrees=True
    )
    hv_model.compute_parameters(memoryview(zeta), memoryview(r0_f32), memoryview(th0_f32))
    r0_f64, th0_f64 = python_f64_hv57(zeta, 21.0, 1.7e-14, 1.55e-6, 25000.0, 4000)

    hv_rel_r0 = np.abs(r0_f32.astype(np.float64) - r0_f64) / r0_f64
    hv_rel_th0 = np.abs(th0_f32.astype(np.float64) - th0_f64) / th0_f64
    max_rel_r0 = float(np.max(hv_rel_r0))
    max_rel_th0 = float(np.max(hv_rel_th0))

    print(f"[2] HV-5/7 Optical Turbulence (N = {N:,}, f32 SIMD vs f64):")
    print(f"    Max. relative error (r_0)     : {max_rel_r0:.6e}  -> {format_latex_sci(max_rel_r0)}")
    print(f"    Max. relative error (theta_0) : {max_rel_th0:.6e}")

    # --------------------------------------------------------------------------
    # 3 & 4. Wideband 64-QAM Signal, Physical PA, GMP NMSE, and 3-iter ILA ACPR
    # --------------------------------------------------------------------------
    x_ref = allocate_aligned(N, np.complex64)
    x_ref[:] = generate_rrc_64qam_signal(N, sps=8, beta=0.22, seed=2026)

    y_pa_unlin = allocate_aligned(N, np.complex64)
    y_pa_unlin[:] = simulate_physical_pa(x_ref)

    dpd = phy_engine.DPD_GMP_Model(K=5, M=3, L=1)
    P = dpd.num_coefficients

    # Metric 3: NMSE between nonlinear PA output and forward GMP model output
    w_fwd = allocate_aligned(P, np.complex64)
    y_hat = allocate_aligned(N, np.complex64)
    dpd.extract_coefficients(memoryview(y_pa_unlin), memoryview(x_ref), memoryview(w_fwd))
    dpd.apply_dpd(memoryview(x_ref), memoryview(w_fwd), memoryview(y_hat))
    nmse_model_db = compute_nmse_db(y_pa_unlin, y_hat)

    # Metric 4: Closed-loop ILA DPD for n = 3 iterations
    # Gain normalization G_lin matches the PA operating gain at ~1.3 dB compression
    G_lin = 0.86
    u_dpd = allocate_aligned(N, np.complex64)
    y_loop = allocate_aligned(N, np.complex64)
    w_dpd = allocate_aligned(P, np.complex64)
    w_dpd[:] = 0.0 + 0.0j
    w_dpd[0] = 1.0 + 0.0j

    u_dpd[:] = x_ref
    acpr_in = compute_acpr_dbc(x_ref)
    acpr_before = compute_acpr_dbc(y_pa_unlin)

    print(f"[3] Identified GMP Behavioral Model (K=5, M=3, L=1, P={P}, N={N:,}):")
    print(f"    NMSE(y_pa, U(x)*w_model)   : {nmse_model_db:.2f} dB")

    print(f"[4] Closed-Loop ILA DPD (n = 3 iterations, 8x-oversampled 64-QAM):")
    print(f"    Input signal ACPR floor    : L={acpr_in[0]:.2f} dBc, U={acpr_in[1]:.2f} dBc, worst={acpr_in[2]:.2f} dBc")
    print(f"    PA without DPD ACPR        : L={acpr_before[0]:.2f} dBc, U={acpr_before[1]:.2f} dBc, worst={acpr_before[2]:.2f} dBc")

    for it in range(1, 4):
        dpd.apply_dpd(memoryview(x_ref), memoryview(w_dpd), memoryview(u_dpd))
        y_loop[:] = simulate_physical_pa(u_dpd) / np.float32(G_lin)
        # ILA post-inverse training: map y_loop -> u_dpd
        dpd.extract_coefficients(memoryview(u_dpd), memoryview(y_loop), memoryview(w_dpd))

        # Evaluate linearized PA output with updated weights
        dpd.apply_dpd(memoryview(x_ref), memoryview(w_dpd), memoryview(u_dpd))
        y_eval = simulate_physical_pa(u_dpd) / np.float32(G_lin)
        acpr_it = compute_acpr_dbc(y_eval)
        nmse_lin = compute_nmse_db(x_ref, y_eval)
        print(
            f"    ILA Iter {it}: worst ACPR = {acpr_it[2]:.2f} dBc "
            f"(L={acpr_it[0]:.2f}, U={acpr_it[1]:.2f}), PA-out NMSE = {nmse_lin:.2f} dB"
        )

    acpr_after = compute_acpr_dbc(y_eval)
    acpr_improvement_worst = acpr_before[2] - acpr_after[2]
    acpr_improvement_avg = 0.5 * ((acpr_before[0] - acpr_after[0]) + (acpr_before[1] - acpr_after[1]))

    print(f"    ACPR improvement (worst)   : {acpr_improvement_worst:+.2f} dB")
    print(f"    ACPR improvement (avg L/U) : {acpr_improvement_avg:+.2f} dB")

    out = {
        "itu_max_abs_err_db": itu_max_abs_err,
        "itu_max_abs_err_latex": format_latex_sci(itu_max_abs_err, "dB"),
        "hv57_max_rel_err_r0": max_rel_r0,
        "hv57_max_rel_err_r0_latex": format_latex_sci(max_rel_r0),
        "gmp_identified_nmse_db": nmse_model_db,
        "acpr_before_worst_dbc": acpr_before[2],
        "acpr_after_worst_dbc": acpr_after[2],
        "acpr_improvement_worst_db": acpr_improvement_worst,
        "acpr_improvement_avg_db": acpr_improvement_avg,
    }
    print("\nJSON_ACCURACY:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
