/**
 * AudioFeedbackPanel
 *
 * Controls for the adaptive audio neurofeedback system.
 * Rendered as a card on the Live tab.
 *
 * Three tones fired by useAudioFeedback:
 *   🔔 Focus Bell    (220 Hz sine)  — fires when focus_score drops below threshold
 *   ✨ EA-1 Chime    (528 Hz)       — fires on EA-1 eligibility entry
 *   🧠 Wander Pulse  (110 Hz)       — fires on engagement_index spike
 */
import React from 'react'
import type { AudioFeedbackReturn, AudioSensitivity } from '../hooks/useAudioFeedback'

interface Props {
  audio: AudioFeedbackReturn
}

const s: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', gap: 14 },
  row:  { display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' as const },
  label: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase' as const, letterSpacing: 0.8, minWidth: 80,
  },
  toggleBtn: (on: boolean): React.CSSProperties => ({
    padding: '6px 16px', borderRadius: 20, fontSize: 13, fontWeight: 700,
    cursor: 'pointer',
    border: `1px solid ${on ? '#238636' : '#30363d'}`,
    background: on ? 'rgba(46,160,67,0.15)' : 'rgba(139,148,158,0.1)',
    color: on ? '#3fb950' : '#8b949e',
    transition: 'all 180ms ease',
    userSelect: 'none' as const,
  }),
  sensitivityBtn: (active: boolean): React.CSSProperties => ({
    padding: '4px 12px', borderRadius: 6, fontSize: 12, fontWeight: 600,
    cursor: 'pointer',
    border: `1px solid ${active ? '#388bfd' : '#30363d'}`,
    background: active ? 'rgba(56,139,253,0.12)' : 'transparent',
    color: active ? '#58a6ff' : '#8b949e',
    transition: 'all 150ms ease',
  }),
  slider: {
    flex: 1, minWidth: 100, maxWidth: 160,
    accentColor: '#388bfd', cursor: 'pointer',
  },
  eventLog: {
    fontSize: 12, color: '#8b949e', fontStyle: 'italic',
    padding: '6px 10px',
    background: 'rgba(22,27,34,0.6)',
    border: '1px solid #21262d',
    borderRadius: 6,
    minHeight: 32,
  },
  toneList: {
    display: 'flex', gap: 8, flexWrap: 'wrap' as const,
  },
  tonePill: {
    fontSize: 11, padding: '2px 8px', borderRadius: 20,
    border: '1px solid #21262d', color: '#8b949e',
    background: 'rgba(33,38,45,0.6)',
  },
}

const SENSITIVITIES: AudioSensitivity[] = ['low', 'medium', 'high']

export default function AudioFeedbackPanel({ audio }: Props) {
  return (
    <div style={s.root}>
      {/* Enable / disable toggle */}
      <div style={s.row}>
        <span style={s.label}>Audio</span>
        <button style={s.toggleBtn(audio.enabled)} onClick={audio.toggle}>
          {audio.enabled ? '🔊 On' : '🔇 Off'}
        </button>
        <span style={{ fontSize: 12, color: '#484f58' }}>
          {audio.enabled ? 'Tones active' : 'Click to enable audio feedback'}
        </span>
      </div>

      {audio.enabled && (
        <>
          {/* Volume */}
          <div style={s.row}>
            <span style={s.label}>Volume</span>
            <input
              type="range" min={0} max={1} step={0.05}
              value={audio.config.volume}
              style={s.slider}
              onChange={e => audio.setVolume(parseFloat(e.target.value))}
            />
            <span style={{ fontSize: 12, color: '#8b949e', minWidth: 32 }}>
              {Math.round(audio.config.volume * 100)}%
            </span>
          </div>

          {/* Sensitivity */}
          <div style={s.row}>
            <span style={s.label}>Sensitivity</span>
            {SENSITIVITIES.map(s_ => (
              <button
                key={s_}
                style={s.sensitivityBtn(audio.config.sensitivity === s_)}
                onClick={() => audio.setSensitivity(s_)}
              >
                {s_.charAt(0).toUpperCase() + s_.slice(1)}
              </button>
            ))}
          </div>

          {/* Tone legend */}
          <div style={s.toneList}>
            <span style={s.tonePill}>🔔 Focus Bell — 220 Hz</span>
            <span style={s.tonePill}>✨ EA-1 Chime — 528 Hz</span>
            <span style={s.tonePill}>🧠 Wander Pulse — 110 Hz</span>
          </div>

          {/* Last event log */}
          <div style={s.eventLog}>
            {audio.lastEvent ?? 'No events yet this session'}
          </div>
        </>
      )}
    </div>
  )
}
