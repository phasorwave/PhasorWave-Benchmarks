#!/usr/bin/env node
/**
 * WASM SIMD128 Benchmark Runner for PhasorWave (`phy_engine_wasm.js`).
 * Executes all three full physical-layer kernels at N = 50,000 with 50 warm-up iterations
 * and 100 measured iterations, reporting median, mean, min, and P95 latencies in ms.
 *
 * Requires the benchmark WASM build (`make -f Makefile.wasm wasm-benchmark`) and aborts
 * immediately if Community stub error codes (-1 or -2) or dpd_num_coeffs === 0 are returned.
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const N = 50000;
const WARMUP_ITERS = 50;
const MEASURE_ITERS = 100;

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function statsMs(samplesMs) {
  const sorted = [...samplesMs].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  return {
    median_ms: percentile(sorted, 0.5),
    mean_ms: sum / sorted.length,
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
    p95_ms: percentile(sorted, 0.95),
    runs: sorted.length,
  };
}

// Deterministic PRNG (Mulberry32) so inputs are reproducible without external deps
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}

function boxMullerPair(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const mag = Math.sqrt(-2.0 * Math.log(u1));
  const ang = 2.0 * Math.PI * u2;
  return [mag * Math.cos(ang), mag * Math.sin(ang)];
}

function benchmarkKernel(warmup, runs, fn, validateFn) {
  for (let i = 0; i < warmup; i++) {
    const rc = fn();
    if (validateFn) validateFn(rc);
  }
  const times = new Array(runs);
  for (let i = 0; i < runs; i++) {
    const t0 = process.hrtime.bigint();
    const rc = fn();
    const t1 = process.hrtime.bigint();
    if (validateFn) validateFn(rc);
    times[i] = Number(t1 - t0) / 1e6;
  }
  return statsMs(times);
}

function loadWasmFactory(wasmJsPath) {
  const code = fs.readFileSync(wasmJsPath, 'utf8');
  const modObj = { exports: {} };
  const wrapper = new Function(
    'exports',
    'require',
    'module',
    '__filename',
    '__dirname',
    code
  );
  wrapper(modObj.exports, require, modObj, wasmJsPath, path.dirname(wasmJsPath));
  return modObj.exports;
}

function resolveWasmJsPath() {
  const cliArg = process.argv.find((a) => a.endsWith('.js') && a !== __filename);
  const wasmArgFlagIdx = process.argv.indexOf('--wasm');
  if (wasmArgFlagIdx !== -1 && process.argv[wasmArgFlagIdx + 1]) {
    return path.resolve(process.cwd(), process.argv[wasmArgFlagIdx + 1]);
  }
  if (cliArg) {
    return path.resolve(process.cwd(), cliArg);
  }
  if (process.env.PHASORWAVE_WASM_JS) {
    return path.resolve(process.cwd(), process.env.PHASORWAVE_WASM_JS);
  }
  return path.resolve(__dirname, '..', 'build', 'wasm-benchmark', 'phy_engine_wasm.js');
}

async function main() {
  const wasmJsPath = resolveWasmJsPath();
  assert.ok(
    fs.existsSync(wasmJsPath),
    `Benchmark WASM bundle not found at ${wasmJsPath}. Build it first with: make -f Makefile.wasm wasm-benchmark`
  );
  const wasmBinPath = wasmJsPath.replace(/\.js$/i, '.wasm');
  assert.ok(
    fs.existsSync(wasmBinPath),
    `Benchmark WASM binary not found at ${wasmBinPath}. Build it first with: make -f Makefile.wasm wasm-benchmark`
  );
  const wasmSizeBytes = fs.statSync(wasmBinPath).size;

  const createPhyEngineModule = loadWasmFactory(wasmJsPath);
  const mod = await createPhyEngineModule();

  // --------------------------------------------------------------------------
  // 1. ITU-R P.618-14 (N = 50,000 link geometries)
  // --------------------------------------------------------------------------
  const fGhz = 20.0;
  const kCoeff = 0.0951;
  const alphaCoeff = 1.0080;
  const pTarget = 0.05;

  const pR = mod._wasm_alloc_f32(N);
  const pHs = mod._wasm_alloc_f32(N);
  const pTheta = mod._wasm_alloc_f32(N);
  const pPhi = mod._wasm_alloc_f32(N);
  const pOutAp = mod._wasm_alloc_f32(N);

  {
    const rng = makeRng(61814);
    const heap = mod.HEAPF32;
    const oR = pR >>> 2;
    const oHs = pHs >>> 2;
    const oTh = pTheta >>> 2;
    const oPh = pPhi >>> 2;
    for (let i = 0; i < N; i++) {
      heap[oR + i] = 5.0 + 90.0 * rng();
      heap[oHs + i] = 3.2 * rng();
      heap[oTh + i] = 5.0 + 80.0 * rng();
      heap[oPh + i] = -60.0 + 120.0 * rng();
    }
  }

  const assertItuOk = (rc) => {
    assert.notEqual(
      rc,
      -1,
      'ITU-R P.618-14 WASM kernel returned -1 (Community stub blocked). Compile with `make -f Makefile.wasm wasm-benchmark`.'
    );
    assert.notEqual(
      rc,
      -2,
      'ITU-R P.618-14 WASM kernel returned -2 (tier restricted).'
    );
    assert.equal(rc, 0, `ITU-R P.618-14 WASM kernel failed with status ${rc}`);
  };

  const ituStats = benchmarkKernel(
    WARMUP_ITERS,
    MEASURE_ITERS,
    () => mod._wasm_itu_p618_compute(pR, pHs, pTheta, pPhi, pOutAp, N, fGhz, kCoeff, alphaCoeff, pTarget),
    assertItuOk
  );
  const ituSampleSlice = mod.HEAPF32.subarray(pOutAp >>> 2, (pOutAp >>> 2) + 16);
  assert.ok(
    ituSampleSlice.every((v) => Number.isFinite(v) && v >= 0) && Math.max(...ituSampleSlice) > 0,
    `Unexpected ITU-R P.618-14 output slice: [${Array.from(ituSampleSlice).join(', ')}]`
  );

  mod._wasm_free(pR);
  mod._wasm_free(pHs);
  mod._wasm_free(pTheta);
  mod._wasm_free(pPhi);
  mod._wasm_free(pOutAp);

  // --------------------------------------------------------------------------
  // 2. Optical Turbulence HV-5/7 (N = 50,000 zenith angles, 4,000 Simpson steps)
  // --------------------------------------------------------------------------
  const vWind = 21.0;
  const aGround = 1.7e-14;
  const wavelength = 1.55e-6;
  const hMax = 25000.0;
  const numSteps = 4000;

  const pZeta = mod._wasm_alloc_f32(N);
  const pR0 = mod._wasm_alloc_f32(N);
  const pTheta0 = mod._wasm_alloc_f32(N);

  {
    const rng = makeRng(1550);
    const heap = mod.HEAPF32;
    const oZ = pZeta >>> 2;
    for (let i = 0; i < N; i++) {
      heap[oZ + i] = 75.0 * rng();
    }
  }

  const assertHvOk = (rc) => {
    assert.notEqual(
      rc,
      -1,
      'HV-5/7 WASM kernel returned -1 (Community stub blocked). Compile with `make -f Makefile.wasm wasm-benchmark`.'
    );
    assert.notEqual(
      rc,
      -2,
      'HV-5/7 WASM kernel returned -2 (tier restricted).'
    );
    assert.equal(rc, 0, `HV-5/7 WASM kernel failed with status ${rc}`);
  };

  const hvStats = benchmarkKernel(
    WARMUP_ITERS,
    MEASURE_ITERS,
    () => mod._wasm_optical_hv57_compute(pZeta, pR0, pTheta0, N, vWind, aGround, wavelength, hMax, numSteps),
    assertHvOk
  );
  const sampleR0 = mod.HEAPF32[pR0 >>> 2];
  const sampleTh0 = mod.HEAPF32[pTheta0 >>> 2];
  assert.ok(Number.isFinite(sampleR0) && sampleR0 > 0, `Unexpected HV-5/7 r0 sample: ${sampleR0}`);
  assert.ok(Number.isFinite(sampleTh0) && sampleTh0 > 0, `Unexpected HV-5/7 theta0 sample: ${sampleTh0}`);

  mod._wasm_free(pZeta);
  mod._wasm_free(pR0);
  mod._wasm_free(pTheta0);

  // --------------------------------------------------------------------------
  // 3. ILA-GMP MGS-QR (N = 50,000 samples, K=5, M=3, L=1 -> P = 39)
  // Note: `_wasm_dpd_gmp_extract_and_apply` runs QR extraction + 1 GMP apply.
  // --------------------------------------------------------------------------
  const K = 5;
  const M = 3;
  const L = 1;
  const P = mod._wasm_dpd_gmp_num_coeffs(K, M, L);
  assert.ok(
    Number.isFinite(P) && P > 0,
    `dpd_num_coeffs returned ${P} (Community tier cap active). Compile with \`make -f Makefile.wasm wasm-benchmark\`.`
  );
  assert.equal(P, 39, `Expected P = 39 for GMP (K=${K}, M=${M}, L=${L}), got ${P}`);

  const pX = mod._wasm_alloc_f32(2 * N);
  const pY = mod._wasm_alloc_f32(2 * N);
  const pW = mod._wasm_alloc_f32(2 * P);
  const pLin = mod._wasm_alloc_f32(2 * N);

  {
    const rng = makeRng(2026);
    const heap = mod.HEAPF32;
    const oX = pX >>> 2;
    const oY = pY >>> 2;
    for (let i = 0; i < N; i++) {
      const [gI, gQ] = boxMullerPair(rng);
      let xi = 0.32 * gI;
      let xq = 0.32 * gQ;
      const mag = Math.hypot(xi, xq);
      if (mag > 0.85) {
        const s = 0.85 / mag;
        xi *= s;
        xq *= s;
      }
      heap[oX + 2 * i] = xi;
      heap[oX + 2 * i + 1] = xq;

      const r2 = xi * xi + xq * xq;
      const r4 = r2 * r2;
      const gre = 1.0 - 0.18 * r2 - 0.06 * r4;
      const gim = 0.07 * r2 - 0.03 * r4;
      heap[oY + 2 * i] = xi * gre - xq * gim;
      heap[oY + 2 * i + 1] = xi * gim + xq * gre;
    }
  }

  let flops = 0;
  const assertDpdOk = (rc) => {
    assert.notEqual(
      rc,
      -1,
      'ILA-GMP WASM kernel returned -1 (invalid args or error).'
    );
    assert.notEqual(
      rc,
      -2,
      'ILA-GMP WASM kernel returned -2 (Community tier restriction blocked GMP K=5,M=3,L=1). Compile with `make -f Makefile.wasm wasm-benchmark`.'
    );
    assert.ok(Number.isFinite(rc) && rc > 0, `Expected positive FLOPs from ILA-GMP kernel, got ${rc}`);
    flops = rc;
  };

  const dpdStats = benchmarkKernel(
    WARMUP_ITERS,
    MEASURE_ITERS,
    () => mod._wasm_dpd_gmp_extract_and_apply(pX, pY, pW, pLin, N, K, M, L),
    assertDpdOk
  );

  mod._wasm_free(pX);
  mod._wasm_free(pY);
  mod._wasm_free(pW);
  mod._wasm_free(pLin);

  const result = {
    runtime: `Node.js ${process.version} (V8 WebAssembly SIMD128)`,
    wasm_binary_path: wasmBinPath,
    wasm_size_bytes: wasmSizeBytes,
    batch_size: N,
    warmup_iters: WARMUP_ITERS,
    measure_iters: MEASURE_ITERS,
    dpd_num_coeffs: P,
    dpd_ila_flops: flops,
    kernels: {
      itu_p618: ituStats,
      dpd_ila_qr: dpdStats,
      hv57_optical: hvStats,
    },
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
