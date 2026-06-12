/**
 * RollingSpectrogram — Canvas 2D real-time STFT spectrogram.
 *
 * Layout: one canvas per Muse S channel (TP9, AF7, AF8, TP10) plus a
 * channel-averaged view selectable via tabs.  Each frame the DFT of the
 * most recent eeg_samples window is computed, power is log-scaled, and a
 * new column is painted on the right edge while the existing image scrolls
 * one pixel left.
 *
 * Frequency axis: 1–50 Hz, logarithmic, with band boundary lines.
 * Colour: diverging RdBu centred on the running session mean.
 * Update rate: driven by SSE frames (~4 Hz).
 */

import React, { useRef, useEffect, useState, useCallback } from 'react'

// ── constants ─────────────────────────────────────────────────────────────────
const CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'] as const
type Channel = typeof CHANNELS[number] | 'Mean'
const ALL_CHANNELS: Channel[] = [...CHANNELS, 'Mean']

const CANVAS_W = 480
const CANVAS_H = 120
const FREQ_BINS = 64          // DFT output bins we visualise
const MAX_FREQ  = 50          // Hz
const MIN_FREQ  = 1           // Hz
const HISTORY   = CANVAS_W    // one pixel per frame column

// Band boundary frequencies (Hz)
const BAND_LINES = [
  { hz: 4,  label: 'δ/θ', color: 'rgba(255,255,255,0.25)' },
  { hz: 8,  label: 'θ/α', color: 'rgba(255,255,255,0.35)' },
  { hz: 13, label: 'α/β', color: 'rgba(255,255,255,0.35)' },
  { hz: 30, label: 'β/γ', color: 'rgba(255,255,255,0.20)' },
]

// ── DFT helpers ───────────────────────────────────────────────────────────────
function hann(n: number, N: number): number {
  return 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)))
}

/** Compute power spectrum (magnitude²) for a real signal. */
function powerSpectrum(samples: number[], sampleRate = 256): Float32Array {
  const N = samples.length
  // Apply Hann window
  const windowed = samples.map((v, i) => v * hann(i, N))
  // DFT — O(N²) but N is small (≤256 samples)
  const half = Math.floor(N / 2)
  const power = new Float32Array(half)
  for (let k = 0; k < half; k++) {
    let re = 0, im = 0
    const omega = (2 * Math.PI * k) / N
    for (let n = 0; n < N; n++) {
      re += windowed[n] * Math.cos(omega * n)
      im -= windowed[n] * Math.sin(omega * n)
    }
    power[k] = re * re + im * im
  }
  return power
}

/** Map a frequency (Hz) to a canvas y-pixel (log scale, 0=top). */
function freqToY(hz: number, h: number): number {
  const logMin = Math.log(MIN_FREQ)
  const logMax = Math.log(MAX_FREQ)
  const logHz  = Math.log(Math.max(MIN_FREQ, Math.min(hz, MAX_FREQ)))
  return h - ((logHz - logMin) / (logMax - logMin)) * h
}

/** Extract the power at a specific frequency bin. */
function binForFreq(hz: number, N: number, sr: number): number {
  return Math.round((hz * N) / sr)
}

// ── RdBu diverging colormap ───────────────────────────────────────────────────
// Colour centres on 0 deviation from mean; negative=blue, positive=red.
function rdbu(t: number): [number, number, number] {
  // t in [-1, 1]
  const v = Math.max(-1, Math.min(1, t))
  if (v < 0) {
    // blue side: white → blue
    const s = -v
    const r = Math.round(255 * (1 - s * 0.78))
    const g = Math.round(255 * (1 - s * 0.58))
    const b = 255
    return [r, g, b]
  } else {
    // red side: white → red
    const r = 255
    const g = Math.round(255 * (1 - v * 0.75))
    const bl = Math.round(255 * (1 - v * 0.75))
    return [r, g, bl]
  }
}

// ── Props ─────────────────────────────────────────────────────────────────────
export interface Props {
  /** Raw EEG samples: shape [numChannels][numSamples].  Index 0=TP9,1=AF7,2=AF8,3=TP10. */
  eegSamples: number[][] | null
  sampleRate?: number
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function RollingSpectrogram({ eegSamples, sampleRate = 256 }: Props) {
  const [activeChannel, setActiveChannel] = useState<Channel>('Mean')
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  // Rolling history: array of FREQ_BINS-length columns
  const historyRef   = useRef<Float32Array[]>([])
  // Running mean for normalisation
  const meanRef      = useRef<Float32Array>(new Float32Array(FREQ_BINS))
  const countRef     = useRef(0)

  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const history = historyRef.current
    const mean    = meanRef.current
    const W = CANVAS_W, H = CANVAS_H

    ctx.clearRect(0, 0, W, H)
    const imgData = ctx.createImageData(W, H)
    const data = imgData.data

    for (let col = 0; col < history.length; col++) {
      const pwr = history[col]
      const x = W - history.length + col
      if (x < 0) continue

      for (let y = 0; y < H; y++) {
        // Map pixel y → frequency
        const logMin = Math.log(MIN_FREQ)
        const logMax = Math.log(MAX_FREQ)
        const hz = Math.exp(logMin + ((H - y) / H) * (logMax - logMin))
        const bin = binForFreq(hz, pwr.length * 2, sampleRate)
        const clampedBin = Math.max(0, Math.min(bin, pwr.length - 1))

        const rawPwr = pwr[clampedBin]
        const mu     = mean[clampedBin] || 1e-10
        const dev    = (rawPwr - mu) / (mu + 1e-10)  // normalised deviation
        const t      = Math.max(-1, Math.min(1, dev * 2))
        const [r, g, b] = rdbu(t)

        const idx = (y * W + x) * 4
        data[idx]     = r
        data[idx + 1] = g
        data[idx + 2] = b
        data[idx + 3] = 255
      }
    }
    ctx.putImageData(imgData, 0, 0)

    // Band boundary lines
    ctx.save()
    BAND_LINES.forEach(({ hz, label, color }) => {
      const y = freqToY(hz, H)
      ctx.strokeStyle = color
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(W, y)
      ctx.stroke()
      ctx.fillStyle = 'rgba(255,255,255,0.5)'
      ctx.font = '9px monospace'
      ctx.fillText(label, 4, y - 2)
    })
    ctx.restore()

    // Frequency axis labels
    ctx.save()
    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.font = '9px monospace'
    ;[2, 4, 8, 13, 20, 30, 50].forEach(hz => {
      const y = freqToY(hz, H)
      ctx.fillText(`${hz}`, W - 22, y + 3)
    })
    ctx.restore()
  }, [sampleRate])

  useEffect(() => {
    if (!eegSamples) return

    // Build per-channel power, then pick active
    const perChannel: Float32Array[] = eegSamples.map(ch => powerSpectrum(ch, sampleRate))

    let col: Float32Array
    if (activeChannel === 'Mean') {
      col = new Float32Array(perChannel[0]?.length ?? 0)
      perChannel.forEach(ch => ch.forEach((v, i) => { col[i] += v }))
      col.forEach((_, i) => { col[i] /= perChannel.length })
    } else {
      const idx = CHANNELS.indexOf(activeChannel as typeof CHANNELS[number])
      col = perChannel[idx] ?? new Float32Array(0)
    }

    // Update running mean (exponential moving average)
    const mean = meanRef.current
    const alpha = 0.05
    col.forEach((v, i) => { mean[i] = mean[i] * (1 - alpha) + v * alpha })
    countRef.current++

    // Append column and trim to HISTORY length
    historyRef.current.push(col)
    if (historyRef.current.length > HISTORY) historyRef.current.shift()

    paint()
  }, [eegSamples, activeChannel, paint, sampleRate])

  // Repaint when channel tab changes
  useEffect(() => { paint() }, [activeChannel, paint])

  const tabStyle = (ch: Channel): React.CSSProperties => ({
    padding: '3px 10px',
    fontSize: 11,
    fontWeight: 600,
    borderRadius: 4,
    border: 'none',
    cursor: 'pointer',
    background: ch === activeChannel ? '#388bfd' : 'transparent',
    color: ch === activeChannel ? '#fff' : '#8b949e',
    transition: 'background 0.15s',
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {ALL_CHANNELS.map(ch => (
          <button key={ch} style={tabStyle(ch)} onClick={() => {
            historyRef.current = []
            meanRef.current = new Float32Array(FREQ_BINS)
            setActiveChannel(ch)
          }}>{ch}</button>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{ width: '100%', height: 'auto', borderRadius: 6, imageRendering: 'pixelated' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#484f58' }}>
        <span>1 Hz</span>
        <span style={{ color: '#8b949e', fontSize: 11 }}>Frequency (log) · RdBu centred on session mean</span>
        <span>50 Hz</span>
      </div>
      {!eegSamples && (
        <div style={{ fontSize: 12, color: '#484f58', textAlign: 'center', padding: '20px 0' }}>
          Waiting for EEG stream…
        </div>
      )}
    </div>
  )
}
