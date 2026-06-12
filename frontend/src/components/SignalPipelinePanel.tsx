/**
 * SignalPipelinePanel
 *
 * A unified read-only status panel that visualises the live state of the
 * entire Stage 0 → 3 DSP pipeline in one card.
 *
 * Layout (top-to-bottom):
 *   Stage 0 — Raw acquisition  : sample rate + eeg_samples shape
 *   Stage 1 — Filtering        : high-pass + notch config (static labels)
 *   Stage 2 — Channel quality  : bad_channels pill list (live)
 *   Stage 3 — Artifact gate    : current frame decision + rolling rate bar
 *
 * This gives a single-glance health summary for the signal pipeline
 * directly on the Live tab without navigating away.
 */
import React from 'react'
import type { NeurolinkState } from '../types'
import type { ArtifactStats }  from '../hooks/useArtifactStats'

interface Props {
  state:    NeurolinkState | null
  stats:    ArtifactStats
}

const MUSE_CH = ['TP9', 'AF7', 'AF8', 'TP10']

// ── Helpers ──────────────────────────────────────────────────────────────────
function stageRowStyle(ok: boolean): React.CSSProperties {
  return {
    display: 'flex', alignItems: 'flex-start', gap: 10,
    padding: '10px 12px', borderRadius: 8,
    background: ok ? 'rgba(22,27,34,0.6)' : 'rgba(248,81,73,0.06)',
    border: `1px solid ${ok ? '#30363d' : 'rgba(248,81,73,0.25)'}`,
    transition: 'background 0.3s ease, border-color 0.3s ease',
  }
}

function StageLabel({ n, label }: { n: string; label: string }) {
  return (
    <div style={{ minWidth: 68, flexShrink: 0 }}>
      <span style={{
        display: 'inline-block',
        fontSize: 10, fontWeight: 700, color: '#484f58',
        textTransform: 'uppercase', letterSpacing: 0.8,
        background: '#21262d', border: '1px solid #30363d',
        borderRadius: 4, padding: '1px 6px', marginBottom: 4,
      }}>
        Stage {n}
      </span>
      <div style={{ fontSize: 11, fontWeight: 600, color: '#8b949e' }}>{label}</div>
    </div>
  )
}

function Dot({ ok }: { ok: boolean }) {
  return (
    <span style={{
      marginTop: 2, width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
      background: ok ? '#3fb950' : '#f85149',
      transition: 'background 0.3s ease',
    }} />
  )
}

function MiniBar({ rate, colour }: { rate: number; colour: string }) {
  return (
    <div style={{ height: 5, background: '#21262d', borderRadius: 3, overflow: 'hidden', width: '100%', marginTop: 5 }}>
      <div style={{
        height: '100%',
        width: `${Math.min(rate * 100, 100)}%`,
        background: colour,
        borderRadius: 3,
        transition: 'width 0.3s ease',
      }} />
    </div>
  )
}

const CAUSE_LABELS: Record<string, string> = {
  amplitude: 'Amp',
  motion:    'Motion',
  kurtosis:  'Kurtosis',
}

function rateColour(r: number) {
  if (r >= 0.30) return '#f85149'
  if (r >= 0.10) return '#e3b341'
  return '#3fb950'
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SignalPipelinePanel({ state, stats }: Props) {
  const connected    = state?.connected ?? false
  const badChannels  = state?.bad_channels ?? []
  const rejected     = state?.artifact_rejected ?? false
  const reasons      = state?.artifact_reasons  ?? []
  const eegSamples   = state?.eeg_samples ?? []
  const nCh          = eegSamples.length
  const nSamp        = eegSamples[0]?.length ?? 0
  const frameCount   = state?.frame_count ?? 0
  const rc           = rateColour(stats.rejectRate)

  const stage0ok = connected && nCh > 0 && nSamp > 0
  const stage2ok = badChannels.length === 0
  const stage3ok = !rejected

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

      {/* Stage 0 — Raw acquisition */}
      <div style={stageRowStyle(stage0ok)}>
        <Dot ok={stage0ok} />
        <StageLabel n="0" label="Acquisition" />
        <div style={{ fontSize: 12, color: '#cdd9e5', flex: 1 }}>
          {connected ? (
            nCh > 0
              ? <>{nCh} ch · {nSamp} samp/frame · Frame #{frameCount}</>
              : <span style={{ color: '#484f58' }}>Waiting for EEG samples…</span>
          ) : (
            <span style={{ color: '#484f58' }}>Not connected</span>
          )}
        </div>
      </div>

      {/* Stage 1 — Filtering (static config labels — backend-only step) */}
      <div style={stageRowStyle(true)}>
        <Dot ok={true} />
        <StageLabel n="1" label="Filtering" />
        <div style={{ fontSize: 12, color: '#8b949e', flex: 1, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span style={filterPill}>HP 0.5 Hz</span>
          <span style={filterPill}>Notch 50/60 Hz</span>
          <span style={filterPill}>Zero-phase FIR</span>
        </div>
      </div>

      {/* Stage 2 — Bad channel detection + interpolation */}
      <div style={stageRowStyle(stage2ok)}>
        <Dot ok={stage2ok} />
        <StageLabel n="2" label="Channels" />
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {MUSE_CH.map(ch => {
              const bad = badChannels.map(c => c.toUpperCase()).includes(ch)
              return (
                <span key={ch} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                  background: bad ? 'rgba(248,81,73,0.12)' : 'rgba(46,160,67,0.1)',
                  border: `1px solid ${bad ? 'rgba(248,81,73,0.3)' : 'rgba(46,160,67,0.25)'}`,
                  color: bad ? '#f85149' : '#3fb950',
                }}>
                  <span style={{
                    width: 5, height: 5, borderRadius: '50%',
                    background: bad ? '#f85149' : '#3fb950', flexShrink: 0,
                  }} />
                  {ch}
                  {bad && <span style={{ fontSize: 10, opacity: 0.7 }}>↯</span>}
                </span>
              )
            })}
          </div>
          {badChannels.length > 0 && (
            <div style={{ fontSize: 11, color: '#e3b341', marginTop: 5 }}>
              ↻ Spherical-spline interpolation applied
            </div>
          )}
        </div>
      </div>

      {/* Stage 3 — Artifact gate */}
      <div style={stageRowStyle(stage3ok)}>
        <Dot ok={stage3ok} />
        <StageLabel n="3" label="Artifact Gate" />
        <div style={{ flex: 1 }}>
          {/* Current frame decision */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{
              fontSize: 12, fontWeight: 600,
              color: stage3ok ? '#3fb950' : '#e3b341',
            }}>
              {stage3ok ? 'Frame accepted' : 'Frame rejected'}
            </span>
            {reasons.length > 0 && (
              <span style={{ fontSize: 11, color: '#8b949e' }}>
                ({reasons.map(r => CAUSE_LABELS[r] ?? r).join(' · ')})
              </span>
            )}
          </div>
          {/* Rolling rate mini-bar */}
          <div style={{ fontSize: 11, color: '#484f58', display: 'flex', justifyContent: 'space-between' }}>
            <span>Reject rate (last {stats.windowSize} frames)</span>
            <span style={{ color: rc }}>{(stats.rejectRate * 100).toFixed(1)}%</span>
          </div>
          <MiniBar rate={stats.rejectRate} colour={rc} />
        </div>
      </div>
    </div>
  )
}

const filterPill: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center',
  padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
  background: 'rgba(88,166,255,0.08)',
  border: '1px solid rgba(88,166,255,0.2)',
  color: '#58a6ff',
}
