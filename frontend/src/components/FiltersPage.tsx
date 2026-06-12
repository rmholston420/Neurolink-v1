import React from 'react'
import { useFilters, FilterToggles } from '../hooks/useFilters'

interface Props {
  apiUrl: string
}

type FilterKey = keyof FilterToggles

interface FilterMeta {
  key: FilterKey
  label: string
  stage: string
  description: string
  warnOnDisable?: string
}

const FILTERS: FilterMeta[] = [
  {
    key: 'stage1_fir',
    label: 'FIR Filter Chain',
    stage: 'Stage 1',
    description:
      'Zero-phase high-pass (0.5 Hz), notch (50/60 Hz + harmonic), and low-pass (45 Hz) FIR filters. ' +
      'Removes electrode drift, power-line interference, and high-frequency muscle noise before any downstream processing.',
    warnOnDisable: 'Raw unfiltered EEG will reach all downstream stages — band powers and artifact detection will be unreliable.',
  },
  {
    key: 'stage2_bad_channels',
    label: 'Bad Channel Detection',
    stage: 'Stage 2',
    description:
      'EMA-based variance and PSD outlier detection. Channels whose signal statistics deviate significantly from neighbours ' +
      'are flagged and reconstructed via spherical-spline interpolation so artifact energy does not contaminate adjacent channels.',
  },
  {
    key: 'stage3_artifact_gate',
    label: 'Artifact Gate',
    stage: 'Stage 3',
    description:
      'Epoch-level rejection gate running three independent checks per frame: amplitude threshold (±100 µV), ' +
      'excess kurtosis burst detection (> 5.0), and IMU motion RMS threshold (0.15 g). ' +
      'Rejected frames are excluded from band powers, ASR, and regression.',
    warnOnDisable: 'All frames — including those with eye blinks, EMG bursts, and movement artifacts — will reach band power computation.',
  },
  {
    key: 'imu_gate',
    label: 'IMU Motion Gate',
    stage: 'Stage 0 / 3',
    description:
      'Accelerometer-based motion gating. Uses the IMU RMS threshold to flag frames contaminated by head movement or body sway. ' +
      'Shared between Stage 0 (acquisition readiness) and Stage 3 (per-frame motion criterion).',
  },
  {
    key: 'stage4_asr',
    label: 'Artifact Subspace Reconstruction (ASR)',
    stage: 'Stage 4',
    description:
      'Reconstructs EEG bursts that exceed the calibration covariance by more than 20 SDs (BurstCriterion). ' +
      'Preferred over ICA for low-channel-count wearable EEG. Calibrates automatically during the session baseline window.',
  },
  {
    key: 'stage4b_baseline',
    label: 'Session Baseline Recorder',
    stage: 'Stage 4b',
    description:
      '150-second eyes-closed resting baseline. First 30 s discarded for dry-electrode impedance stabilisation; ' +
      'remaining 120 s feed ASR covariance calibration. Fires a bell notification when complete.',
    warnOnDisable: 'ASR will calibrate on regular session data rather than a dedicated clean resting window.',
  },
  {
    key: 'stage5_ocular',
    label: 'Ocular Regression (Gratton–Coles)',
    stage: 'Stage 5',
    description:
      'Gratton–Coles temporal regression removes eye-blink and eye-movement artifacts using the AUX/EOG channel as reference. ' +
      'OLS regression coefficients are refitted every ~2 minutes to track slow skin-potential drift. ' +
      'Falls back to pass-through when no AUX channel is present.',
  },
]

// ── Styles ──────────────────────────────────────────────────────────────────
const S: Record<string, React.CSSProperties> = {
  page: { display: 'flex', flexDirection: 'column', gap: 14 },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 4,
  },
  pageTitle: { fontSize: 13, fontWeight: 700, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
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
  right: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0, paddingTop: 2 },
  badge: {
    fontSize: 10, fontWeight: 700, letterSpacing: 0.5, padding: '2px 8px',
    borderRadius: 10,
  },
  // Toggle switch
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
  loading: { color: '#8b949e', fontSize: 13, padding: 20 },
  errorBox: {
    background: 'rgba(248,81,73,0.08)', border: '1px solid rgba(248,81,73,0.3)',
    borderRadius: 8, padding: '10px 14px', fontSize: 12, color: '#f85149',
  },
}

function ToggleSwitch({ enabled, onToggle, ariaLabel }: { enabled: boolean; onToggle: () => void; ariaLabel: string }) {
  return (
    <button
      role="switch"
      aria-checked={enabled}
      aria-label={ariaLabel}
      onClick={onToggle}
      style={{
        ...S.switchTrack,
        background: enabled ? '#1f6feb' : '#30363d',
      }}
    >
      <span style={{ ...S.switchThumb, left: enabled ? 21 : 3 }} />
    </button>
  )
}

function FilterRow({ meta, enabled, onToggle }: { meta: FilterMeta; enabled: boolean; onToggle: () => void }) {
  return (
    <div style={{ ...S.card, ...(enabled ? {} : S.cardDisabled) }}>
      <div style={S.left}>
        <span style={S.stageTag}>{meta.stage}</span>
        <div style={S.filterLabel}>{meta.label}</div>
        <div style={S.filterDesc}>{meta.description}</div>
        {!enabled && meta.warnOnDisable && (
          <div style={S.warn}>
            <span>⚠</span>
            <span>{meta.warnOnDisable}</span>
          </div>
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
          onToggle={onToggle}
          ariaLabel={`${enabled ? 'Disable' : 'Enable'} ${meta.label}`}
        />
      </div>
    </div>
  )
}

export default function FiltersPage({ apiUrl }: Props) {
  const { toggles, loading, error, toggle, resetAll } = useFilters(apiUrl)

  if (loading) return <div style={S.loading}>Loading filter state…</div>

  return (
    <div style={S.page}>
      <div style={S.header}>
        <div style={S.pageTitle}>Pipeline Stage Filters</div>
        <button
          style={S.resetBtn}
          onClick={resetAll}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#cdd9e5'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#8b949e' }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#8b949e'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#30363d' }}
        >
          Reset to defaults
        </button>
      </div>

      {error && <div style={S.errorBox}>⚠ Could not reach filter API: {error}</div>}

      {FILTERS.map(meta => (
        <FilterRow
          key={meta.key}
          meta={meta}
          enabled={toggles[meta.key]}
          onToggle={() => toggle(meta.key)}
        />
      ))}
    </div>
  )
}
