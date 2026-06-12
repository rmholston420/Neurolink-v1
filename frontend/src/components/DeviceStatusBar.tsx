/**
 * DeviceStatusBar
 *
 * Compact always-visible strip showing:
 *   • Battery level  — segmented bar + numeric %
 *   • Signal quality — 4-bar icon derived from contact_quality (0–1)
 *   • Source badge   — adapter name when connected
 *
 * Works for both Path A (Web Bluetooth — battery from useMuseBLE) and
 * Path B (Backend BLE — contact_quality from NeurolinkState SSE stream).
 *
 * Props:
 *   battery        number | null   0-100 from useMuseBLE; null on Path B
 *   contactQuality number | null   0-1 from NeurolinkState; null until first frame
 *   poorContact    boolean         fallback binary flag when contactQuality is null
 *   source         string | null   state.source, e.g. "muse_s" / "mock"
 *   connected      boolean
 */
import React from 'react'

interface Props {
  battery:        number | null
  contactQuality: number | null
  poorContact:    boolean
  source:         string | null
  connected:      boolean
}

// ── Colour helpers ────────────────────────────────────────────────────────────
function batteryColour(pct: number): string {
  if (pct <= 15) return '#f85149'   // red
  if (pct <= 35) return '#e3b341'   // amber
  return '#3fb950'                  // green
}

function signalColour(q: number): string {
  if (q >= 0.75) return '#3fb950'
  if (q >= 0.40) return '#e3b341'
  return '#f85149'
}

// Resolve a 0–1 quality score from whichever signal we have
function resolveQuality(contactQuality: number | null, poorContact: boolean): number | null {
  if (contactQuality !== null) return contactQuality
  if (poorContact) return 0.1   // binary "bad"
  return null                   // no data yet
}

// ── Battery bar ──────────────────────────────────────────────────────────────
function BatteryBar({ pct }: { pct: number | null }) {
  const SEGMENTS = 5
  const critical = pct !== null && pct <= 15

  const containerStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
  }

  const barWrapStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 2,
    padding: '2px 3px',
    border: '1px solid rgba(139,148,158,0.3)',
    borderRadius: 4,
    position: 'relative',
  }

  // Battery tip
  const tipStyle: React.CSSProperties = {
    width: 3,
    height: 6,
    background: 'rgba(139,148,158,0.4)',
    borderRadius: '0 2px 2px 0',
    flexShrink: 0,
  }

  const labelStyle: React.CSSProperties = {
    fontSize: 11,
    fontWeight: 600,
    color: pct !== null ? batteryColour(pct) : '#484f58',
    letterSpacing: 0.3,
    minWidth: 30,
    animation: critical ? 'nlPulse 1.2s ease-in-out infinite' : 'none',
  }

  const filled = pct !== null ? Math.round((pct / 100) * SEGMENTS) : 0
  const colour = pct !== null ? batteryColour(pct) : '#484f58'

  return (
    <div style={containerStyle} title={pct !== null ? `Battery: ${pct}%` : 'Battery level unavailable'}>
      <div style={barWrapStyle}>
        {Array.from({ length: SEGMENTS }, (_, i) => (
          <div
            key={i}
            style={{
              width: 5,
              height: 11,
              borderRadius: 2,
              background: i < filled ? colour : 'rgba(139,148,158,0.15)',
              transition: 'background 400ms ease',
            }}
          />
        ))}
      </div>
      <div style={tipStyle} />
      <span style={labelStyle}>
        {pct !== null ? `${pct}%` : '—'}
      </span>
    </div>
  )
}

// ── Signal bars ───────────────────────────────────────────────────────────────
function SignalBars({ quality }: { quality: number | null }) {
  const BARS = 4
  // Map 0–1 quality to filled bars count (0–4)
  const filled = quality === null ? 0 : Math.max(1, Math.round(quality * BARS))
  const colour = quality !== null ? signalColour(quality) : '#484f58'
  const label  = quality !== null
    ? quality >= 0.75 ? 'Good' : quality >= 0.40 ? 'Fair' : 'Poor'
    : 'No data'
  const tooltip = quality !== null
    ? `Signal quality: ${label} (${Math.round(quality * 100)}%)`
    : 'Signal quality unavailable'

  const containerStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'flex-end',
    gap: 2,
  }

  const textStyle: React.CSSProperties = {
    fontSize: 11,
    fontWeight: 600,
    color: colour,
    marginLeft: 5,
    letterSpacing: 0.2,
  }

  return (
    <div
      style={{ display: 'inline-flex', alignItems: 'center', gap: 0 }}
      title={tooltip}
    >
      <div style={containerStyle}>
        {Array.from({ length: BARS }, (_, i) => {
          const barH = 5 + i * 3   // 5px, 8px, 11px, 14px
          const active = quality !== null && i < filled
          return (
            <div
              key={i}
              style={{
                width: 4,
                height: barH,
                borderRadius: 2,
                background: active ? colour : 'rgba(139,148,158,0.18)',
                alignSelf: 'flex-end',
                transition: 'background 400ms ease',
              }}
            />
          )
        })}
      </div>
      <span style={textStyle}>{label}</span>
    </div>
  )
}

// ── Source badge ──────────────────────────────────────────────────────────────
function SourceBadge({ source, connected }: { source: string | null; connected: boolean }) {
  if (!connected || !source) return null
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      padding: '2px 8px',
      borderRadius: 20,
      fontSize: 11,
      fontWeight: 600,
      background: 'rgba(88,166,255,0.1)',
      border: '1px solid rgba(88,166,255,0.25)',
      color: '#58a6ff',
      letterSpacing: 0.3,
    }}>
      {source}
    </span>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function DeviceStatusBar({
  battery, contactQuality, poorContact, source, connected,
}: Props) {
  const quality = resolveQuality(contactQuality, poorContact)

  const wrapStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 14,
    padding: '5px 12px',
    background: 'rgba(22,27,34,0.7)',
    border: '1px solid #30363d',
    borderRadius: 8,
    backdropFilter: 'blur(4px)',
  }

  const dividerStyle: React.CSSProperties = {
    width: 1,
    height: 14,
    background: '#30363d',
    flexShrink: 0,
  }

  return (
    <>
      {/* Keyframe for critical battery pulse */}
      <style>{`
        @keyframes nlPulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.35; }
        }
      `}</style>

      <div style={wrapStyle}>
        <BatteryBar pct={battery} />
        <div style={dividerStyle} />
        <SignalBars quality={quality} />
        {source && connected && (
          <>
            <div style={dividerStyle} />
            <SourceBadge source={source} connected={connected} />
          </>
        )}
      </div>
    </>
  )
}
