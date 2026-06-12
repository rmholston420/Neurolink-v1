import React from 'react'
import { useArtifactConfig, ArtifactGateConfig } from '../hooks/useArtifactConfig'

interface Props {
  apiUrl: string
}

// ── Threshold slider metadata ────────────────────────────────────────────────
interface ThresholdMeta {
  key: keyof Pick<ArtifactGateConfig, 'pk2pk_uv' | 'accel_rms_g' | 'kurtosis_threshold'>
  enableKey: keyof Pick<ArtifactGateConfig, 'enable_amplitude' | 'enable_imu' | 'enable_kurtosis'>
  label: string
  unit: string
  min: number
  max: number
  step: number
  description: string
  warnOnDisable?: string
}

const THRESHOLDS: ThresholdMeta[] = [
  {
    key: 'pk2pk_uv',
    enableKey: 'enable_amplitude',
    label: 'Amplitude Gate',
    unit: 'µV',
    min: 50,
    max: 500,
    step: 10,
    description:
      'Peak-to-peak amplitude threshold per frame. Frames where any channel ' +
      'exceeds ±(threshold/2) µV are rejected. Default 100 µV. ' +
      'Lower values reject more aggressively; raise to 200+ µV only if clean signal ' +
      'is routinely discarded during vigorous meditation.',
    warnOnDisable:
      'Frames with extreme voltage spikes (electrode pops, body movement) will ' +
      'reach band-power computation unfiltered.',
  },
  {
    key: 'accel_rms_g',
    enableKey: 'enable_imu',
    label: 'IMU Motion Gate',
    unit: 'g RMS',
    min: 0.01,
    max: 1.0,
    step: 0.01,
    description:
      'Accelerometer RMS threshold for motion-gated frame rejection. ' +
      'Frames where head / device motion exceeds this value are discarded before ' +
      'band-power analysis. Default 0.15 g. Reduce to 0.05 g for breath-only sessions; ' +
      'raise to 0.30 g for walking meditation.',
    warnOnDisable:
      'Head movement and body sway artifacts will contaminate band powers, ' +
      'invalidating delta and theta readings.',
  },
  {
    key: 'kurtosis_threshold',
    enableKey: 'enable_kurtosis',
    label: 'Kurtosis Burst Gate',
    unit: 'k',
    min: 2.0,
    max: 20.0,
    step: 0.5,
    description:
      'Excess kurtosis threshold for EMG / transient burst detection. ' +
      'A Gaussian signal has kurtosis ≈ 3; EMG bursts and electrode pops produce ' +
      'kurtosis > 5. Default 5.0. Lower to 3.5 for high-sensitivity rejection; ' +
      'raise to 8.0 if genuine gamma activity is being over-rejected.',
    warnOnDisable:
      'Jaw clenches, muscle artifacts, and electrode pop transients will pass ' +
      'through to gamma-band computation undetected.',
  },
]

// ── Styles (identical token values to FiltersPage) ────────────────────────────
const S: Record<string, React.CSSProperties> = {
  section: { display: 'flex', flexDirection: 'column', gap: 14 },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 4,
  },
  sectionTitle: {
    fontSize: 13, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase', letterSpacing: 1,
  },
  resetBtn: {
    padding: '5px 14px', fontSize: 12, fontWeight: 600,
    background: 'none', border: '1px solid #30363d',
    borderRadius: 6, color: '#8b949e', cursor: 'pointer',
    transition: 'border-color 0.15s, color 0.15s',
  },
  card: {
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 10, padding: '16px 20px',
    display: 'flex', alignItems: 'flex-start', gap: 16,
  },
  cardDisabled: {
    background: '#0d1117', border: '1px solid #21262d',
    opacity: 0.75,
  },
  left: { flex: 1 },
  stageTag: {
    display: 'inline-block',
    fontSize: 10, fontWeight: 700, color: '#388bfd',
    background: 'rgba(56,139,253,0.1)', border: '1px solid rgba(56,139,253,0.3)',
    borderRadius: 4, padding: '1px 6px', marginBottom: 6, letterSpacing: 0.5,
  },
  filterLabel: { fontSize: 14, fontWeight: 700, color: '#cdd9e5', marginBottom: 4 },
  filterDesc: { fontSize: 12, color: '#8b949e', lineHeight: 1.55, maxWidth: 640 },
  warn: {
    display: 'flex', alignItems: 'center', gap: 5,
    fontSize: 11, color: '#e3b341', marginTop: 6,
  },
  right: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    gap: 8, flexShrink: 0, paddingTop: 2,
  },
  badge: {
    fontSize: 10, fontWeight: 700, letterSpacing: 0.5, padding: '2px 8px',
    borderRadius: 10,
  },
  switchTrack: {
    width: 40, height: 22, borderRadius: 11, border: 'none',
    cursor: 'pointer', transition: 'background 0.2s',
    position: 'relative', flexShrink: 0, padding: 0,
  },
  switchThumb: {
    position: 'absolute', top: 3, width: 16, height: 16,
    borderRadius: '50%', background: '#fff',
    transition: 'left 0.2s',
    pointerEvents: 'none',
  },
  numRow: {
    display: 'flex', alignItems: 'center', gap: 6, marginTop: 10,
  },
  numBtn: {
    width: 26, height: 26, borderRadius: 6, border: '1px solid #30363d',
    background: 'none', color: '#8b949e', cursor: 'pointer',
    fontSize: 16, lineHeight: '1', display: 'flex', alignItems: 'center',
    justifyContent: 'center', flexShrink: 0,
    transition: 'border-color 0.15s, color 0.15s',
  },
  numDisplay: {
    minWidth: 70, textAlign: 'center', fontSize: 13, fontWeight: 700,
    color: '#cdd9e5', background: '#0d1117',
    border: '1px solid #30363d', borderRadius: 6,
    padding: '4px 8px',
  },
  numUnit: { fontSize: 11, color: '#8b949e' },
  loading: { color: '#8b949e', fontSize: 13, padding: 20 },
  errorBox: {
    background: 'rgba(248,81,73,0.08)', border: '1px solid rgba(248,81,73,0.3)',
    borderRadius: 8, padding: '10px 14px', fontSize: 12, color: '#f85149',
  },
}

// ── Sub-components ────────────────────────────────────────────────────────────
function ToggleSwitch({ enabled, onToggle, ariaLabel }: {
  enabled: boolean; onToggle: () => void; ariaLabel: string
}) {
  return (
    <button
      role="switch"
      aria-checked={enabled}
      aria-label={ariaLabel}
      onClick={onToggle}
      style={{ ...S.switchTrack, background: enabled ? '#1f6feb' : '#30363d' }}
    >
      <span style={{ ...S.switchThumb, left: enabled ? 21 : 3 }} />
    </button>
  )
}

function NumericControl({ value, unit, min, max, step, onChange }: {
  value: number; unit: string; min: number; max: number; step: number
  onChange: (v: number) => void
}) {
  const dec = () => onChange(Math.max(min, parseFloat((value - step).toFixed(10))))
  const inc = () => onChange(Math.min(max, parseFloat((value + step).toFixed(10))))
  const display = Number.isInteger(step) ? value.toFixed(0) : value.toFixed(2)
  return (
    <div style={S.numRow}>
      <button
        style={S.numBtn}
        onClick={dec}
        aria-label={`Decrease by ${step}`}
        onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#cdd9e5'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#8b949e' }}
        onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#8b949e'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#30363d' }}
      >−</button>
      <span style={S.numDisplay}>{display}</span>
      <span style={S.numUnit}>{unit}</span>
      <button
        style={S.numBtn}
        onClick={inc}
        aria-label={`Increase by ${step}`}
        onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#cdd9e5'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#8b949e' }}
        onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#8b949e'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#30363d' }}
      >+</button>
    </div>
  )
}

function ThresholdRow({
  meta, config, onUpdate,
}: {
  meta: ThresholdMeta
  config: ArtifactGateConfig
  onUpdate: (patch: Partial<ArtifactGateConfig>) => void
}) {
  const enabled = config[meta.enableKey] as boolean
  const value   = config[meta.key] as number

  return (
    <div style={{ ...S.card, ...(enabled ? {} : S.cardDisabled) }}>
      <div style={S.left}>
        <span style={S.stageTag}>Stage 3</span>
        <div style={S.filterLabel}>{meta.label}</div>
        <div style={S.filterDesc}>{meta.description}</div>
        {!enabled && meta.warnOnDisable && (
          <div style={S.warn}>
            <span>⚠</span>
            <span>{meta.warnOnDisable}</span>
          </div>
        )}
        {enabled && (
          <NumericControl
            value={value}
            unit={meta.unit}
            min={meta.min}
            max={meta.max}
            step={meta.step}
            onChange={v => onUpdate({ [meta.key]: v })}
          />
        )}
      </div>
      <div style={S.right}>
        <span style={{
          ...S.badge,
          background: enabled ? 'rgba(46,160,67,0.12)' : 'rgba(248,81,73,0.08)',
          color: enabled ? '#3fb950' : '#f85149',
          border: `1px solid ${enabled ? '#238636' : '#da3633'}`,
        }}>
          {enabled ? 'ON' : 'OFF'}
        </span>
        <ToggleSwitch
          enabled={enabled}
          onToggle={() => onUpdate({ [meta.enableKey]: !enabled })}
          ariaLabel={`${enabled ? 'Disable' : 'Enable'} ${meta.label}`}
        />
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function ArtifactConfigPanel({ apiUrl }: Props) {
  const { config, loading, error, updateConfig, resetStats } = useArtifactConfig(apiUrl)

  if (loading) return <div style={S.loading}>Loading Stage 3 config…</div>

  const DEFAULTS: ArtifactGateConfig = {
    pk2pk_uv: 100,
    accel_rms_g: 0.15,
    kurtosis_threshold: 5.0,
    enable_amplitude: true,
    enable_imu: true,
    enable_kurtosis: true,
  }

  return (
    <div style={S.section}>
      <div style={S.header}>
        <div style={S.sectionTitle}>Stage 3 · Artifact Gate Thresholds</div>
        <button
          style={S.resetBtn}
          onClick={() => updateConfig(DEFAULTS)}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#cdd9e5'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#8b949e' }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#8b949e'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#30363d' }}
        >
          Reset to defaults
        </button>
      </div>

      {error && <div style={S.errorBox}>⚠ Could not reach Stage 3 config API: {error}</div>}

      {config && THRESHOLDS.map(meta => (
        <ThresholdRow
          key={meta.key}
          meta={meta}
          config={config}
          onUpdate={updateConfig}
        />
      ))}

      {/* Reset session frame counters */}
      <div style={{
        display: 'flex', justifyContent: 'flex-end', marginTop: 4,
      }}>
        <button
          style={{ ...S.resetBtn, color: '#58a6ff', borderColor: 'rgba(56,139,253,0.4)' }}
          onClick={resetStats}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = '#388bfd' }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(56,139,253,0.4)' }}
        >
          ↺ Reset session frame counters
        </button>
      </div>
    </div>
  )
}
