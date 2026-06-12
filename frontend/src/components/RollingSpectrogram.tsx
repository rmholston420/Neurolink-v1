/**
 * RollingSpectrogram — Canvas 2D rolling STFT spectrogram.
 *
 * Data source priority:
 *   1. Real eeg_samples from SSE frame (4 ch × N samples from hardware adapter)
 *   2. Synthetic signal reconstructed from band powers (offline/mock fallback)
 *
 * The synthetic fallback produces a plausible-looking spectrogram so the
 * visualisation is always live, even when the backend is in mock mode without
 * raw sample emission.
 */

import React, { useEffect, useRef, useCallback } from 'react';
import type { NeurolinkState } from '../types';

const CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'] as const;
type Channel = typeof CHANNELS[number] | 'mean';

const CANVAS_W = 480;
const CANVAS_H = 120;
const FFT_SIZE = 64;
const FS = 256;
const FREQ_MIN = 1;
const FREQ_MAX = 50;
// Band boundary frequencies (Hz) for overlay lines
const BAND_BOUNDARIES = [4, 8, 13, 30];

// Approximate EMA of log power — persists across renders via module-level ref
const _ema: Record<string, number[]> = {};
const EMA_ALPHA = 0.05;

// ── Hann window ─────────────────────────────────────────────────────────────
function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

// ── Radix-2 DFT (real input, power spectrum) ────────────────────────────────
function powerSpectrum(samples: number[], hann: Float32Array): Float32Array {
  const n = hann.length;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) re[i] = (samples[i] ?? 0) * hann[i];

  // DFT O(n²) — adequate for n=64
  const half = n / 2;
  const ps = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    let sumRe = 0, sumIm = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      sumRe += re[t] * Math.cos(angle);
      sumIm -= re[t] * Math.sin(angle);
    }
    ps[k] = sumRe * sumRe + sumIm * sumIm;
  }
  return ps;
}

// ── Map linear frequency bin index → log-scale canvas y ─────────────────────
function freqToY(hz: number, h: number): number {
  const logMin = Math.log10(FREQ_MIN);
  const logMax = Math.log10(FREQ_MAX);
  const norm = (Math.log10(Math.max(hz, FREQ_MIN)) - logMin) / (logMax - logMin);
  return h - norm * h; // flip: high freq at top
}

// ── Diverging RdBu colour: deviation from EMA baseline ─────────────────────
function deviationColour(logP: number, baseline: number): string {
  const dev = logP - baseline;
  const norm = Math.max(-1, Math.min(1, dev / 3)); // ±3 nats = full saturation
  if (norm > 0) {
    const r = Math.round(220 + 35 * norm);
    const g = Math.round(220 - 120 * norm);
    const b = Math.round(220 - 180 * norm);
    return `rgb(${r},${g},${b})`;
  } else {
    const abs = -norm;
    const r = Math.round(220 - 180 * abs);
    const g = Math.round(220 - 80 * abs);
    const b = Math.round(220 + 35 * abs);
    return `rgb(${r},${g},${b})`;
  }
}

// ── Synthetic signal from band powers (mock/offline fallback) ───────────────
function syntheticSamples(bands: NeurolinkState['bands'], n: number, t: number): number[] {
  const out = new Array<number>(n).fill(0);
  const specs: Array<[number, number]> = [
    [2, bands.delta],
    [6, bands.theta],
    [10, bands.alpha],
    [20, bands.beta],
    [40, bands.gamma],
  ];
  for (let i = 0; i < n; i++) {
    const ti = (t * n + i) / FS;
    for (const [f, amp] of specs) {
      out[i] += Math.sqrt(amp + 1e-6) * Math.sin(2 * Math.PI * f * ti);
    }
  }
  return out;
}

// ── Main component ───────────────────────────────────────────────────────────
interface Props {
  state: NeurolinkState | null;
  channel?: Channel;
}

export default function RollingSpectrogram({ state, channel = 'mean' }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  const hannRef = useRef<Float32Array>(hannWindow(FFT_SIZE));

  const draw = useCallback(() => {
    if (!state) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ── Resolve signal for this channel ──────────────────────────────────
    const real = state.eeg_samples;
    const hasReal = Array.isArray(real) && real.length >= 4 && real[0].length >= FFT_SIZE;

    let signal: number[];
    if (hasReal) {
      const chIdx = channel === 'mean'
        ? null
        : CHANNELS.indexOf(channel as typeof CHANNELS[number]);
      if (chIdx === null) {
        // Average across channels
        signal = real[0].map((_, i) =>
          real.reduce((sum, ch) => sum + (ch[i] ?? 0), 0) / real.length
        );
      } else {
        signal = real[chIdx] ?? real[0];
      }
    } else {
      // Synthetic fallback — reconstructed from band powers
      signal = syntheticSamples(state.bands, FFT_SIZE, frameRef.current);
    }

    // Take last FFT_SIZE samples
    const window = signal.slice(-FFT_SIZE);
    const ps = powerSpectrum(window, hannRef.current);
    const logPs = ps.map(p => Math.log(p + 1e-9));

    // ── Update EMA baseline ───────────────────────────────────────────────
    const key = channel;
    if (!_ema[key] || _ema[key].length !== logPs.length) {
      _ema[key] = [...logPs];
    } else {
      for (let i = 0; i < logPs.length; i++) {
        _ema[key][i] = EMA_ALPHA * logPs[i] + (1 - EMA_ALPHA) * _ema[key][i];
      }
    }

    // ── Scroll canvas left by 1px ─────────────────────────────────────────
    const img = ctx.getImageData(1, 0, CANVAS_W - 1, CANVAS_H);
    ctx.putImageData(img, 0, 0);
    ctx.clearRect(CANVAS_W - 1, 0, 1, CANVAS_H);

    // ── Paint new column ──────────────────────────────────────────────────
    const half = ps.length;
    for (let k = 1; k < half; k++) {
      const hz = (k * FS) / (FFT_SIZE * 2);
      if (hz < FREQ_MIN || hz > FREQ_MAX) continue;
      const y1 = Math.round(freqToY(hz, CANVAS_H));
      const y0 = Math.round(freqToY((k - 1) * FS / (FFT_SIZE * 2), CANVAS_H));
      const h = Math.max(1, Math.abs(y0 - y1));
      ctx.fillStyle = deviationColour(logPs[k], _ema[key][k]);
      ctx.fillRect(CANVAS_W - 1, y1, 1, h);
    }

    // ── Band boundary overlay lines ───────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1;
    for (const hz of BAND_BOUNDARIES) {
      const y = Math.round(freqToY(hz, CANVAS_H));
      ctx.beginPath();
      ctx.moveTo(CANVAS_W - 8, y);
      ctx.lineTo(CANVAS_W - 1, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    frameRef.current += 1;
  }, [state, channel]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div style={{ position: 'relative', background: '#111', borderRadius: 6, overflow: 'hidden' }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{ display: 'block', width: '100%', imageRendering: 'pixelated' }}
      />
      {/* Frequency axis labels */}
      <div style={{
        position: 'absolute', top: 0, left: 4,
        fontSize: 9, color: 'rgba(255,255,255,0.5)',
        display: 'flex', flexDirection: 'column',
        height: CANVAS_H, justifyContent: 'space-between', pointerEvents: 'none'
      }}>
        <span>50Hz</span>
        <span>13Hz</span>
        <span>4Hz</span>
        <span>1Hz</span>
      </div>
      {state && !state.eeg_samples?.length && (
        <div style={{
          position: 'absolute', bottom: 2, right: 4,
          fontSize: 9, color: 'rgba(255,200,80,0.7)',
          pointerEvents: 'none'
        }}>
          synthetic (no raw samples)
        </div>
      )}
    </div>
  );
}
