/**
 * ArtifactGuidePanel  v2
 *
 * Live artifact intelligence panel for Neurolink-v1.
 *
 * Sections:
 *   1. Signal Quality Gauge  (0-100 score, colour-coded)
 *   2. Live Active Artifacts  (heuristic auto-detection from live state)
 *   3. Recommended Pipeline Checklist  (interactive, based on research)
 *   4. Consumer-Grade EEG Caveats  (Muse / dry-electrode notes)
 *   5. EEG Artifact Encyclopedia  (full collapsible reference, 7 types)
 *
 * Detection heuristics (all soft — no hard thresholds that duplicate
 * the backend Stage 3 gates):
 *   - Ocular   : artifact_reasons includes "amplitude"
 *   - Muscle   : artifact_reasons includes "kurtosis"; OR gamma > alpha×4
 *   - Cardiac  : artifact_reasons includes "cardiac"; OR hr_bpm > 0 and
 *                motion_rms is near-zero (unlikely movement, so pulse)
 *   - Movement : artifact_reasons includes "motion"|"imu"; OR motion_rms > 0.15
 *   - Power-line: gamma > beta×6 AND gamma > 8 µV² (crude proxy for 50/60 Hz)
 *   - Electrode pop: poor_contact OR any bad_channels
 *   - Bad channel: bad_channels.length > 0
 */
import React, { useState, useEffect, useRef } from 'react'
import type { NeurolinkState } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ArtifactGuidePanelProps {
  state: NeurolinkState | null
  rejectRate: number
  connected: boolean
}

type ArtifactClass =
  | 'ocular'
  | 'muscle'
  | 'cardiac'
  | 'movement'
  | 'powerline'
  | 'electrode'
  | 'badchannel'

interface ArtifactInfo {
  id: ArtifactClass
  label: string
  icon: string
  colour: string
  borderColour: string
  bgColour: string
  frequencyRange: string
  affectedChannels: string
  waveformCue: string
  identificationCues: string[]
  immediateActions: string[]
  preventionTips: string[]
  signalStageNote: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Artifact Encyclopedia Data  (7 types, research-grounded)
// ─────────────────────────────────────────────────────────────────────────────

const ARTIFACT_ENCYCLOPEDIA: ArtifactInfo[] = [
  {
    id: 'ocular',
    label: 'Ocular (Eye) Artifact',
    icon: '👁',
    colour: '#58a6ff',
    borderColour: 'rgba(88,166,255,0.35)',
    bgColour: 'rgba(88,166,255,0.07)',
    frequencyRange: '0.1 – 10 Hz  (overlaps delta & theta)',
    affectedChannels: 'Fp1, Fp2, AF3, AF4 — frontal electrodes',
    waveformCue:
      'Slow, high-amplitude corneoretinal potentials. Blink: sharp spike with rounded recovery (~200–400 ms). Saccade: slow lateral drift on F7/F8.',
    identificationCues: [
      'Sudden large spike on frontal channels (Fp1/Fp2) lasting ~200–400 ms — classic blink signature',
      'Amplitude commonly 100–500 µV — far exceeds typical EEG of 10–50 µV',
      'Waveform does NOT follow scalp topography — disproportionately frontal',
      'Slow lateral drift on F7/F8 during horizontal eye movements (saccades)',
      '71% of wearable EEG artifact studies identify ocular artifacts as the most prevalent type',
    ],
    immediateActions: [
      'Instruct subject to keep eyes soft-focused or closed during critical recording windows',
      'Stage 3 Amplitude Gate auto-rejects epochs where pk2pk > threshold (default 150 µV)',
      'For open-eye protocols, widen the amplitude gate threshold to avoid rejecting all data',
      'If an EOG reference channel is available, enable Gratton–Coles regression in the backend',
    ],
    preventionTips: [
      'Closed-eye meditation dramatically reduces blink artifact prevalence',
      'Pre-session eye-relaxation exercises can reduce saccade frequency',
      'Good Fp1/Fp2 electrode gel quality is critical — poor contact amplifies ocular spread',
    ],
    signalStageNote:
      'Caught by Stage 3 Amplitude Gate (≤ 150 µV default). ICA-based removal via ICLabel is available in MNE-Python for post-hoc offline analysis. Frequency overlap with delta/theta makes simple high-pass filtering destructive — ICA or regression is required.',
  },
  {
    id: 'muscle',
    label: 'EMG / Muscle Artifact',
    icon: '💪',
    colour: '#d27bff',
    borderColour: 'rgba(210,123,255,0.35)',
    bgColour: 'rgba(210,123,255,0.07)',
    frequencyRange: '20 – 300 Hz  (broadband high-frequency)',
    affectedChannels: 'Temporal (T3/T4/T5/T6) and frontal — wherever muscles underlie electrodes',
    waveformCue:
      'Dense, irregular, high-frequency bursts with no repeating structure — looks like dense "grass" in the raw EEG trace.',
    identificationCues: [
      'Sudden broadband power increase above 20 Hz — especially obvious in 40–80 Hz gamma band',
      'Distribution follows jaw, temple, or neck muscle anatomy — not neural topography',
      'Kurtosis spikes above 5 (leptokurtic) — the Stage 3 Kurtosis Gate catches this',
      'Millisecond-scale spikes cluster into short bursts (50–200 ms) then disappear',
      'Jaw clenching → bilateral temporal contamination; neck tension → posterior channel noise',
    ],
    immediateActions: [
      'Ask subject to relax the jaw — slightly open mouth, tongue resting on lower palate',
      'Shoulder-rolling and neck release for 30 s before session start prevents anticipatory tension',
      'Stage 3 Kurtosis Burst Gate (threshold 5.0 k) rejects most severe bursts automatically',
      'Reduce Kurtosis threshold to 3.5 if gamma band contamination is suspected',
    ],
    preventionTips: [
      'Pre-session progressive muscle relaxation (feet → scalp) is the single most effective prevention',
      'Avoid caffeinated beverages 2 hours before — caffeine increases jaw tension',
      'Ensure the headset is not overtightened — mechanical pressure on temporal muscles causes involuntary tension',
    ],
    signalStageNote:
      'Caught by Stage 3 Kurtosis Gate. Residual muscle noise above 40 Hz is attenuated by the low-pass filter in FiltersPage. For consumer-grade 4-channel EEG, ICA separation of muscle vs. brain is unreliable — prefer ASR + threshold rejection.',
  },
  {
    id: 'cardiac',
    label: 'Cardiac / BCG Artifact',
    icon: '❤️',
    colour: '#f85149',
    borderColour: 'rgba(248,81,73,0.35)',
    bgColour: 'rgba(248,81,73,0.07)',
    frequencyRange: '~1.2 Hz fundamental + harmonics  (1–10 Hz)',
    affectedChannels: 'Temporal arteries (T3/T4), vertex (Cz) in upright subjects',
    waveformCue:
      'Rhythmic sharp deflection occurring once per heartbeat (~0.8–1.2 s period); waveform resembles a QRS complex and is perfectly periodic.',
    identificationCues: [
      'Periodic rhythm with consistent period matching heart rate — compare to hr_bpm field',
      'Amplitude typically 5–30 µV — smaller than ocular but highly consistent',
      'Waveform shape is stereotyped and identical across cycles (unlike EMG bursts)',
      'Increases in amplitude with elevated heart rate (after stress or exercise)',
      'Most visible at temporal channels (T3/T4) near the carotid arteries',
    ],
    immediateActions: [
      'Compare artifact periodicity to the hr_bpm field in HRVPanel — confirmation of cardiac origin',
      'Ensure low electrode impedance on temporal channels — high impedance amplifies cardiac coupling',
      'Average re-referencing (CAR) partially suppresses this artifact during preprocessing',
      'Post-hoc ICA with ICLabel labels this component "Heart" — remove offline in MNE-Python',
    ],
    preventionTips: [
      'Record 2–3 minutes of resting baseline before session onset — allows heart rate to stabilize',
      'Upright seated posture with relaxed shoulders minimises carotid pulse coupling',
      'Avoid recording immediately after cardiovascular exercise',
    ],
    signalStageNote:
      'Falls below the Stage 3 amplitude threshold (~10 µV) so is not automatically rejected. Post-hoc ICA removal in MNE-Python is the standard method. A future enhancement: add an ECG reference channel for online Gratton-style regression.',
  },
  {
    id: 'movement',
    label: 'Movement / Motion Artifact',
    icon: '🏃',
    colour: '#e3b341',
    borderColour: 'rgba(227,179,65,0.35)',
    bgColour: 'rgba(227,179,65,0.07)',
    frequencyRange: '0.1 – 5 Hz  (low-frequency drift and transients)',
    affectedChannels: 'All channels simultaneously for head movement; single channel for cable drag',
    waveformCue:
      'Slow-wave drift with occasional sharp transients when motion starts or stops; amplitude can exceed 500 µV on all channels simultaneously.',
    identificationCues: [
      'IMU motion_rms > 0.15 g — direct physical evidence of head movement',
      'All channels affected simultaneously — distinguishes it from electrode-specific noise',
      'Low-frequency (<3 Hz) large-amplitude waves that are aperiodic',
      'Abrupt onset and offset matching visible body movement events',
      'Cable-drag artifacts appear on individual channels as sharp asymmetric spikes',
    ],
    immediateActions: [
      'Stage 3 IMU Motion Gate (0.15 g) automatically rejects these frames — verify it is enabled in FiltersPage',
      'Check motion_rms in the IMU Panel — values > 0.15 g confirm movement contamination',
      'For walking meditation: raise IMU threshold to 0.30 g to retain usable data',
      'Route all electrode cables along the headband — cable sway is a major localized artifact source',
    ],
    preventionTips: [
      'Secure all cables or use fully wireless streaming to eliminate cable drag',
      'Seated or supine recording minimizes body movement artifacts',
      'Collect a clean resting calibration segment first — ASR uses it as its clean reference baseline',
    ],
    signalStageNote:
      'Caught by Stage 3 IMU Motion Gate. The motion_rms field is streamed live in NeurolinkState. IMU-gated rejection is a significantly underused technique in wearable EEG research — only a minority of published pipelines use it.',
  },
  {
    id: 'powerline',
    label: 'Power-Line Interference',
    icon: '⚡',
    colour: '#3fb950',
    borderColour: 'rgba(63,185,80,0.35)',
    bgColour: 'rgba(63,185,80,0.07)',
    frequencyRange: '50 Hz (Europe/Asia) or 60 Hz (Americas) — perfectly sinusoidal',
    affectedChannels: 'All channels equally (global, constant amplitude)',
    waveformCue:
      'Perfectly sinusoidal constant-amplitude oscillation at exactly 50 or 60 Hz. Appears as a bright, sharply-defined horizontal stripe in the Rolling Spectrogram.',
    identificationCues: [
      'PSD shows a sharp spike at exactly 50 Hz or 60 Hz — far narrower than broadband EMG',
      'Constant amplitude and frequency — unaffected by cognitive state or movement',
      'Visible as a bright horizontal band in RollingSpectrogram at the line frequency',
      'Amplitude increases near poorly shielded power supplies, USB hubs, fluorescent lights',
      'Disappears when the laptop is switched to battery power — confirms mains coupling',
    ],
    immediateActions: [
      'Verify Notch Filter (50/60 Hz) is enabled in FiltersPage — this is the primary and sufficient defense',
      'Move setup away from switching power supplies, USB hubs, and fluorescent lighting',
      'Switch laptop to battery power during critical recording windows',
      'Ensure electrode cables are not routed parallel to power cables',
    ],
    preventionTips: [
      'Battery-powered amplifiers avoid ground loops that couple mains noise',
      'Keep EEG amplifier cable distance from the nearest AC power source as short as possible',
      'Shielded electrode cables reduce electromagnetic induction from nearby power lines',
    ],
    signalStageNote:
      'Handled in Stage 1 by the Notch Filter in FiltersPage. The filter frequency is configurable for 50 Hz (EU) or 60 Hz (US/Canada). A correctly applied narrow notch filter leaves all other frequencies perfectly intact.',
  },
  {
    id: 'electrode',
    label: 'Electrode Pop / Impedance Drift',
    icon: '🔌',
    colour: '#ffa657',
    borderColour: 'rgba(255,166,87,0.35)',
    bgColour: 'rgba(255,166,87,0.07)',
    frequencyRange: '< 0.5 Hz (drift) + wideband transient (pop)',
    affectedChannels: 'Exactly ONE channel — this spatial specificity is the key diagnostic feature',
    waveformCue:
      'A single channel shows an abrupt step-change ("pop") or slow monotonic drift while all other channels remain clean.',
    identificationCues: [
      'Affects exactly ONE channel while all others are normal — critical localisation clue',
      'Electrode pop: sudden large-amplitude step (> 1000 µV possible) followed by slow return',
      'Impedance drift: slow wandering away from zero (DC offset) increasing over minutes',
      'ImpedancePanel shows > 50 kΩ on the affected channel',
      'Sweat under dry electrodes causes gradual impedance reduction but also chemical drift',
    ],
    immediateActions: [
      'Check ImpedancePanel — re-seat the electrode if impedance > 50 kΩ',
      'For gel electrodes: re-apply conductive gel to the affected electrode',
      'For dry electrodes: gentle pressure with small circular motion can restore temporary contact',
      'Stage 2 bad channel detection will flag and exclude the channel automatically',
    ],
    preventionTips: [
      'Verify all channel impedances are < 20 kΩ (green in ImpedancePanel) before every session',
      'For dry electrode systems: ensure scalp is clean — oils and hair products increase contact resistance',
      'For sessions > 30 min: plan a mid-session impedance check to catch gel drying',
    ],
    signalStageNote:
      'The High-Pass Filter (0.5 Hz, Stage 1) attenuates slow drift. Electrode pops are caught by the Stage 3 Amplitude Gate. The Stage 2 Bad Channel pipeline identifies and interpolates affected channels automatically.',
  },
  {
    id: 'badchannel',
    label: 'Bad Channel (Flat / Noisy)',
    icon: '📡',
    colour: '#8b949e',
    borderColour: 'rgba(139,148,158,0.35)',
    bgColour: 'rgba(139,148,158,0.07)',
    frequencyRange: 'Flat: DC (zero signal)  |  Noisy: wideband',
    affectedChannels: 'Any channel — identified by abnormal PSD relative to its neighbours',
    waveformCue:
      'Flat channel: constant zero or near-zero signal. Noisy channel: much higher variance than all others with no coherent spectral pattern.',
    identificationCues: [
      'BadChannelPanel highlights the channel by name — the authoritative real-time source',
      'Flat channel: PSD is orders of magnitude below all other channels across all frequencies',
      'Noisy channel: PSD is uniformly 2–3× above neighbours with no spectral peaks',
      'One channel reads very differently from its spatial neighbours',
      'Contact quality indicator is red or yellow for the affected electrode',
    ],
    immediateActions: [
      'bad_channels field in NeurolinkState is updated every frame by the Stage 2 pipeline',
      'Bad channels are excluded from band-power computation and EA1 scoring automatically',
      'Re-seat the electrode before the next epoch to attempt recovery',
      'If persistent: note the channel name as excluded in session records',
    ],
    preventionTips: [
      'Run a 30-second impedance check before every session — pre-empts bad channel formation',
      'Apply sufficient gel to fill the electrode cup without shorting adjacent channels',
      'For dry electrodes: part hair at each electrode site to ensure scalp contact',
    ],
    signalStageNote:
      'Handled by Stage 2 bad channel detection. The bad_channels string array lists affected channels by name each frame. Spherical spline interpolation is applied in the backend when channel count permits (≥ 6 good channels remaining).',
  },
]

// ─────────────────────────────────────────────────────────────────────────────
// Reason → Class mapping
// ─────────────────────────────────────────────────────────────────────────────

const REASON_TO_CLASS: Record<string, ArtifactClass> = {
  amplitude: 'ocular',
  kurtosis:  'muscle',
  motion:    'movement',
  imu:       'movement',
  cardiac:   'cardiac',
  powerline: 'powerline',
  drift:     'electrode',
  pop:       'electrode',
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal Quality Score
// ─────────────────────────────────────────────────────────────────────────────

function computeSignalQuality(
  rejectRate: number,
  badChannelCount: number,
  motionRms: number | null,
  poorContact: boolean,
): number {
  let score = 100
  score -= rejectRate * 40
  score -= Math.min(badChannelCount * 10, 30)
  if (motionRms !== null && motionRms > 0.05)
    score -= Math.min((motionRms - 0.05) * 40, 15)
  if (poorContact) score -= 15
  return Math.round(Math.max(0, Math.min(100, score)))
}

function qualityLabel(score: number): { label: string; colour: string } {
  if (score >= 85) return { label: 'Excellent', colour: '#3fb950' }
  if (score >= 65) return { label: 'Good',      colour: '#a5d6a7' }
  if (score >= 40) return { label: 'Fair',       colour: '#e3b341' }
  if (score >= 20) return { label: 'Poor',       colour: '#ffa657' }
  return               { label: 'Critical',   colour: '#f85149' }
}

// ─────────────────────────────────────────────────────────────────────────────
// Heuristic artifact detection
// ─────────────────────────────────────────────────────────────────────────────

function detectActiveArtifacts(state: NeurolinkState | null): ArtifactClass[] {
  if (!state) return []
  const active = new Set<ArtifactClass>()

  // From backend Stage 3 reasons
  if (state.artifact_reasons) {
    for (const reason of state.artifact_reasons) {
      const cls = REASON_TO_CLASS[reason.toLowerCase()]
      if (cls) active.add(cls)
    }
  }

  // Heuristic: gamma ≫ alpha → likely muscle contamination
  const bands = state.bands
  if (bands && bands.gamma > bands.alpha * 4 && bands.gamma > 5) {
    active.add('muscle')
  }

  // Heuristic: gamma ≫ beta AND gamma > 8 → power-line proxy
  if (bands && bands.gamma > bands.beta * 6 && bands.gamma > 8) {
    active.add('powerline')
  }

  // Heuristic: rhythmic heartbeat visible when motion is low
  if (
    state.hr_bpm !== null &&
    state.hr_bpm !== undefined &&
    state.hr_bpm > 0 &&
    (state.motion_rms ?? 0) < 0.05 &&
    !state.artifact_rejected
  ) {
    // Cardiac is always latent; only flag if we have a reason or it's high HR
    if (state.hr_bpm > 90) active.add('cardiac')
  }

  // Motion via IMU
  if ((state.motion_rms ?? 0) > 0.15) active.add('movement')

  // Contact issues
  if (state.poor_contact) active.add('electrode')
  if (state.bad_channels && state.bad_channels.length > 0) active.add('badchannel')

  return Array.from(active)
}

// ─────────────────────────────────────────────────────────────────────────────
// PSD Hint Bar  — visual band power overview
// ─────────────────────────────────────────────────────────────────────────────

const BAND_META: { key: keyof NonNullable<NeurolinkState['bands']>; label: string; colour: string }[] = [
  { key: 'delta', label: 'δ 0.5–4',  colour: '#388bfd' },
  { key: 'theta', label: 'θ 4–8',    colour: '#58a6ff' },
  { key: 'alpha', label: 'α 8–13',   colour: '#3fb950' },
  { key: 'beta',  label: 'β 13–30',  colour: '#e3b341' },
  { key: 'gamma', label: 'γ 30–50',  colour: '#d27bff' },
]

function PsdHintBar({ state }: { state: NeurolinkState | null }) {
  const bands = state?.bands
  if (!bands) return null

  const vals = BAND_META.map(m => bands[m.key] as number)
  const maxVal = Math.max(...vals, 1)
  const total  = vals.reduce((a, b) => a + b, 0) || 1

  // Flag anomalies
  const gammaFrac = bands.gamma / total
  const anomaly =
    gammaFrac > 0.35
      ? { msg: 'High γ fraction — check for muscle/EMG or power-line noise', colour: '#d27bff' }
      : bands.delta > bands.alpha * 3
      ? { msg: 'High δ — check for slow drift, electrode movement, or drowsiness', colour: '#388bfd' }
      : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
        {BAND_META.map((m, i) => (
          <div key={m.key} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <div style={{
              width: '100%',
              height: Math.max(4, Math.round((vals[i] / maxVal) * 48)),
              background: m.colour,
              borderRadius: '3px 3px 0 0',
              opacity: 0.85,
              transition: 'height 0.3s ease',
            }} />
            <div style={{ fontSize: 10, color: '#8b949e', whiteSpace: 'nowrap' }}>{m.label}</div>
          </div>
        ))}
      </div>
      {anomaly && (
        <div style={{
          fontSize: 11, padding: '5px 8px', borderRadius: 6,
          background: 'rgba(255,255,255,0.04)',
          border: `1px solid ${anomaly.colour}44`,
          color: anomaly.colour,
        }}>
          ⚠ {anomaly.msg}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Checklist  (interactive)
// ─────────────────────────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  {
    id: 'impedance',
    label: 'Pre-session impedance check',
    detail: 'All channels < 20 kΩ (green in ImpedancePanel). This is the single highest-value pre-recording action.',
  },
  {
    id: 'hpf',
    label: 'High-pass filter enabled (≥ 0.5 Hz)',
    detail: 'Removes electrode drift and body-sway artifacts without destroying neural delta signals. Set in FiltersPage.',
  },
  {
    id: 'notch',
    label: 'Notch filter enabled (50 or 60 Hz)',
    detail: 'Removes power-line interference. Choose 50 Hz (Europe/Asia) or 60 Hz (Americas) in FiltersPage.',
  },
  {
    id: 'asr',
    label: 'ASR / Burst rejection enabled',
    detail: 'Artifact Subspace Reconstruction is the recommended method for wearable EEG. Enabled via Stage 3 in FiltersPage.',
  },
  {
    id: 'imu',
    label: 'IMU motion gate enabled',
    detail: 'Rejects frames where motion_rms > threshold. Critical for Muse — accelerometer data is a high-value underused stream.',
  },
  {
    id: 'baseline',
    label: '30 s clean calibration baseline recorded',
    detail: 'ASR needs a clean reference segment. Run Calibration from the Live tab before starting the session.',
  },
  {
    id: 'posture',
    label: 'Subject seated, jaw relaxed, eyes soft',
    detail: 'Removes the two most common artifact sources (motion + EMG) at the source before any filtering.',
  },
]

function PipelineChecklist() {
  const [checked, setChecked] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(PIPELINE_STEPS.map(s => [s.id, false]))
  )
  const [expanded, setExpanded] = useState<string | null>(null)

  const completedCount = Object.values(checked).filter(Boolean).length
  const total = PIPELINE_STEPS.length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Progress bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 2 }}>
        <div style={{ flex: 1, height: 5, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${(completedCount / total) * 100}%`,
            background: completedCount === total ? '#3fb950' : '#e3b341',
            borderRadius: 3,
            transition: 'width 0.3s ease, background 0.3s',
          }} />
        </div>
        <span style={{ fontSize: 11, color: '#8b949e', whiteSpace: 'nowrap' }}>
          {completedCount}/{total} ready
        </span>
      </div>

      {PIPELINE_STEPS.map(step => {
        const isChecked  = checked[step.id]
        const isExpanded = expanded === step.id
        return (
          <div key={step.id} style={{
            background: isChecked ? 'rgba(63,185,80,0.06)' : '#0d1117',
            border: `1px solid ${isChecked ? 'rgba(63,185,80,0.25)' : '#21262d'}`,
            borderRadius: 8,
            overflow: 'hidden',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '9px 12px', cursor: 'pointer',
            }}
              onClick={() => setExpanded(isExpanded ? null : step.id)}
            >
              <div
                onClick={e => { e.stopPropagation(); setChecked(c => ({ ...c, [step.id]: !c[step.id] })) }}
                style={{
                  width: 16, height: 16, borderRadius: 4, flexShrink: 0,
                  border: `2px solid ${isChecked ? '#3fb950' : '#484f58'}`,
                  background: isChecked ? '#3fb950' : 'transparent',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', transition: 'all 0.15s',
                }}
                role="checkbox"
                aria-checked={isChecked}
                tabIndex={0}
                onKeyDown={e => e.key === ' ' && setChecked(c => ({ ...c, [step.id]: !c[step.id] }))}
              >
                {isChecked && <span style={{ color: '#0d1117', fontSize: 10, fontWeight: 900, lineHeight: 1 }}>✓</span>}
              </div>
              <span style={{
                fontSize: 12, fontWeight: 600, flex: 1,
                color: isChecked ? '#3fb950' : '#cdd9e5',
                textDecoration: isChecked ? 'none' : 'none',
              }}>
                {step.label}
              </span>
              <span style={{ fontSize: 10, color: '#484f58', transition: 'transform 0.2s',
                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
            </div>
            {isExpanded && (
              <div style={{ padding: '0 12px 10px 38px', fontSize: 11, color: '#8b949e', lineHeight: 1.6 }}>
                {step.detail}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Consumer-EEG Caveats  (Muse / dry electrode specific)
// ─────────────────────────────────────────────────────────────────────────────

const CONSUMER_CAVEATS = [
  {
    icon: '📡',
    title: '4-Channel Limitation',
    body: 'With only 4 channels (TP9, AF7, AF8, TP10), ICA cannot reliably separate independent sources. Prefer ASR + threshold-based rejection over pure ICA for this device class.',
  },
  {
    icon: '🔋',
    title: 'Dry Electrode Drift',
    body: 'Dry electrodes show increasing impedance over time as sweat accumulates. Signal quality typically degrades after 20–30 min. Watch ImpedancePanel for creeping kΩ values.',
  },
  {
    icon: '🏃',
    title: 'IMU is Underutilised — Use It',
    body: 'The Muse IMU accelerometer is one of the most powerful artifact detection signals available. The motion_rms field is available in NeurolinkState every frame. Only a small fraction of published wearable EEG studies use it.',
  },
  {
    icon: '⚡',
    title: 'Higher Noise Floor',
    body: 'Consumer-grade amplifiers have a higher noise floor than research-grade systems. Single-trial analysis is unreliable — always aggregate across multiple trials or use rolling windows for robust estimates.',
  },
  {
    icon: '🧠',
    title: 'Gamma Band Caution',
    body: 'On 4-channel dry-electrode EEG, the gamma band (30–50 Hz) is almost always contaminated by EMG from jaw and temple muscles. Treat gamma power as "EMG proxy" rather than neural gamma unless strict muscle relaxation is verified.',
  },
  {
    icon: '📏',
    title: 'Classify Before You Remove',
    body: 'Applying generic artifact removal without first identifying the artifact type risks destroying genuine neural components. Use the Active Artifacts section above to confirm the type before adjusting thresholds.',
  },
]

function ConsumerCaveats() {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          cursor: 'pointer', padding: '10px 14px',
          background: '#161b22', border: '1px solid #30363d', borderRadius: 10,
        }}
        onClick={() => setOpen(o => !o)}
        role="button"
        aria-expanded={open}
        tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setOpen(o => !o)}
      >
        <span style={{ fontSize: 13, fontWeight: 700, color: '#cdd9e5' }}>📱 Consumer-Grade EEG Caveats</span>
        <span style={{ fontSize: 12, color: '#8b949e', transition: 'transform 0.2s',
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
      </div>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 6 }}>
          <div style={{ fontSize: 12, color: '#8b949e', padding: '2px 2px 4px' }}>
            Specific limitations and best practices for Muse / 4-channel dry-electrode EEG systems.
          </div>
          {CONSUMER_CAVEATS.map((c, i) => (
            <div key={i} style={{
              background: '#0d1117', border: '1px solid #21262d',
              borderRadius: 8, padding: '10px 14px',
              display: 'flex', gap: 10,
            }}>
              <span style={{ fontSize: 18, lineHeight: 1, flexShrink: 0 }}>{c.icon}</span>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#cdd9e5', marginBottom: 3 }}>{c.title}</div>
                <div style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.6 }}>{c.body}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal Quality Gauge
// ─────────────────────────────────────────────────────────────────────────────

function SignalQualityGauge({
  state, rejectRate,
}: { state: NeurolinkState | null; rejectRate: number }) {
  const badCount   = state?.bad_channels?.length ?? 0
  const motionRms  = state?.motion_rms ?? null
  const poorContact = state?.poor_contact ?? false
  const score      = computeSignalQuality(rejectRate, badCount, motionRms, poorContact)
  const { label, colour } = qualityLabel(score)

  const issues: string[] = []
  if (rejectRate >= 0.10) issues.push(`${(rejectRate * 100).toFixed(0)}% frames rejected`)
  if (badCount > 0)       issues.push(`${badCount} bad channel${badCount > 1 ? 's' : ''}`)
  if (poorContact)        issues.push('poor electrode contact')
  if (motionRms !== null && motionRms > 0.15) issues.push(`motion ${motionRms.toFixed(2)} g`)

  const desc = issues.length > 0
    ? `Issues: ${issues.join(' · ')}`
    : 'Signal is clean — no active degradation detected'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 16,
      background: '#161b22', border: '1px solid #30363d',
      borderRadius: 10, padding: '14px 18px',
    }}>
      <div style={{ fontSize: 36, fontWeight: 800, lineHeight: 1, color: colour }}>{score}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: colour, marginBottom: 4 }}>
          Signal Quality · {label}
        </div>
        <div style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.5, marginBottom: 6 }}>{desc}</div>
        <div style={{ height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{
            height: '100%', width: `${score}%`, background: colour, borderRadius: 3,
            transition: 'width 0.4s ease, background 0.4s ease',
          }} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// useRecentActiveClasses — hold cards for 3 s after clearing
// ─────────────────────────────────────────────────────────────────────────────

function useRecentActiveClasses(active: ArtifactClass[], holdMs: number): ArtifactClass[] {
  const [displayed, setDisplayed] = useState<ArtifactClass[]>([])
  const timersRef = useRef<Map<ArtifactClass, ReturnType<typeof setTimeout>>>(new Map())

  useEffect(() => {
    const activeSet = new Set(active)
    setDisplayed(prev => {
      const next = new Set(prev)
      for (const cls of active) next.add(cls)
      return Array.from(next)
    })
    setDisplayed(prev => {
      for (const cls of prev) {
        if (!activeSet.has(cls)) {
          if (!timersRef.current.has(cls)) {
            const t = setTimeout(() => {
              setDisplayed(d => d.filter(c => c !== cls))
              timersRef.current.delete(cls)
            }, holdMs)
            timersRef.current.set(cls, t)
          }
        } else {
          const t = timersRef.current.get(cls)
          if (t !== undefined) { clearTimeout(t); timersRef.current.delete(cls) }
        }
      }
      return prev
    })
  }, [active, holdMs])

  return displayed
}

// ─────────────────────────────────────────────────────────────────────────────
// EncyclopediaEntry
// ─────────────────────────────────────────────────────────────────────────────

const SE: Record<string, React.CSSProperties> = {
  encEntry:   { borderRadius: 8, border: '1px solid #21262d', overflow: 'hidden' },
  encHeader:  { display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', cursor: 'pointer', background: '#161b22' },
  encBody:    { padding: '12px 14px', background: '#0d1117', display: 'flex', flexDirection: 'column', gap: 10 },
  rowLabel:   { fontSize: 10, fontWeight: 700, color: '#484f58', textTransform: 'uppercase', letterSpacing: 0.7 },
  rowValue:   { fontSize: 12, color: '#8b949e', lineHeight: 1.5, marginTop: 2 },
  cueItem:    { fontSize: 12, color: '#cdd9e5', lineHeight: 1.5, paddingLeft: 12, position: 'relative' },
  actionItem: { fontSize: 12, color: '#3fb950', lineHeight: 1.5, paddingLeft: 12, position: 'relative' },
  preventItem:{ fontSize: 12, color: '#58a6ff', lineHeight: 1.5, paddingLeft: 12, position: 'relative' },
  stageNote:  { fontSize: 11, color: '#484f58', background: '#0a0d12', border: '1px solid #21262d', borderRadius: 6, padding: '6px 10px', lineHeight: 1.5 },
}

function EncyclopediaEntry({ info }: { info: ArtifactInfo }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={SE.encEntry}>
      <div style={SE.encHeader} onClick={() => setOpen(o => !o)}
        role="button" aria-expanded={open} tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setOpen(o => !o)}>
        <span style={{ fontSize: 16 }}>{info.icon}</span>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#cdd9e5', flex: 1 }}>{info.label}</span>
        <span style={{ fontSize: 11, color: '#8b949e', transition: 'transform 0.2s',
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
      </div>
      {open && (
        <div style={SE.encBody}>
          {([
            ['Frequency Range', info.frequencyRange],
            ['Affected Channels', info.affectedChannels],
            ['Waveform Signature', info.waveformCue],
          ] as [string, string][]).map(([lbl, val]) => (
            <div key={lbl}>
              <div style={SE.rowLabel}>{lbl}</div>
              <div style={SE.rowValue}>{val}</div>
            </div>
          ))}
          <div>
            <div style={SE.rowLabel}>How to Identify</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 4 }}>
              {info.identificationCues.map((c, i) => (
                <div key={i} style={SE.cueItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#484f58' }}>·</span>{c}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div style={SE.rowLabel}>Immediate Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 4 }}>
              {info.immediateActions.map((a, i) => (
                <div key={i} style={SE.actionItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#2ea043' }}>→</span>{a}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div style={SE.rowLabel}>Prevention</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 4 }}>
              {info.preventionTips.map((p, i) => (
                <div key={i} style={SE.preventItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#388bfd' }}>◈</span>{p}
                </div>
              ))}
            </div>
          </div>
          <div style={SE.stageNote}>
            <span style={{ color: '#388bfd', fontWeight: 700 }}>Pipeline note: </span>
            {info.signalStageNote}
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ActiveArtifactCard
// ─────────────────────────────────────────────────────────────────────────────

function ActiveArtifactCard({ info, isLive }: { info: ArtifactInfo; isLive: boolean }) {
  return (
    <div style={{
      borderRadius: 10, padding: '14px 18px',
      display: 'flex', flexDirection: 'column', gap: 10,
      background: info.bgColour,
      border: `1px solid ${info.borderColour}`,
      opacity: isLive ? 1 : 0.6,
      transition: 'opacity 0.4s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: 22, lineHeight: 1 }}>{info.icon}</span>
        <span style={{ fontSize: 14, fontWeight: 700, color: '#cdd9e5', flex: 1 }}>{info.label}</span>
        {isLive && (
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: 0.5, padding: '2px 8px',
            borderRadius: 10, background: 'rgba(248,81,73,0.15)',
            border: '1px solid rgba(248,81,73,0.4)', color: '#f85149',
          }}>LIVE</span>
        )}
        {!isLive && (
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: 0.5, padding: '2px 8px',
            borderRadius: 10, background: 'rgba(139,148,158,0.1)',
            border: '1px solid rgba(139,148,158,0.2)', color: '#8b949e',
          }}>CLEARED</span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <span style={{
          fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6,
          background: 'rgba(139,148,158,0.1)', border: '1px solid rgba(139,148,158,0.2)', color: '#8b949e',
        }}>⌖ {info.frequencyRange}</span>
        <span style={{
          fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6,
          background: 'rgba(139,148,158,0.1)', border: '1px solid rgba(139,148,158,0.2)', color: '#8b949e',
        }}>⊙ {info.affectedChannels}</span>
      </div>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: '#8b949e',
          textTransform: 'uppercase', letterSpacing: 0.7, marginBottom: 4 }}>How to identify</div>
        {info.identificationCues.slice(0, 3).map((c, i) => (
          <div key={i} style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5,
            paddingLeft: 14, position: 'relative', marginBottom: 3 }}>
            <span style={{ position: 'absolute', left: 0, color: '#484f58' }}>·</span>{c}
          </div>
        ))}
      </div>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: '#8b949e',
          textTransform: 'uppercase', letterSpacing: 0.7, marginBottom: 4 }}>Immediate actions</div>
        {info.immediateActions.slice(0, 3).map((a, i) => (
          <div key={i} style={{ fontSize: 12, color: '#3fb950', lineHeight: 1.5,
            paddingLeft: 14, position: 'relative', marginBottom: 3 }}>
            <span style={{ position: 'absolute', left: 0, color: '#2ea043' }}>→</span>{a}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11, color: '#484f58', background: '#0d1117',
        border: '1px solid #21262d', borderRadius: 6, padding: '6px 10px', lineHeight: 1.5 }}>
        <span style={{ color: '#388bfd', fontWeight: 700 }}>Pipeline: </span>
        {info.signalStageNote}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Export
// ─────────────────────────────────────────────────────────────────────────────

export default function ArtifactGuidePanel({
  state,
  rejectRate,
  connected,
}: ArtifactGuidePanelProps) {
  const [encyclopediaOpen, setEncyclopediaOpen] = useState(false)

  const activeClasses   = detectActiveArtifacts(state)
  const displayedClasses = useRecentActiveClasses(activeClasses, 3000)
  const activeSet       = new Set(activeClasses)

  const activeInfos = displayedClasses
    .map(cls => ARTIFACT_ENCYCLOPEDIA.find(e => e.id === cls))
    .filter((e): e is ArtifactInfo => e !== undefined)

  if (!connected) {
    return (
      <div style={{ color: '#484f58', fontSize: 13, padding: '8px 0' }}>
        Connect a device to enable live artifact guidance.
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* ── 1. Signal Quality Gauge ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#8b949e',
          textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Signal Quality</div>
        <SignalQualityGauge state={state} rejectRate={rejectRate} />
      </div>

      {/* ── 2. Band Power Hint Bar ── */}
      {state?.bands && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#8b949e',
            textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Live Band Power Overview</div>
          <div style={{
            background: '#161b22', border: '1px solid #30363d',
            borderRadius: 10, padding: '14px 18px',
          }}>
            <PsdHintBar state={state} />
          </div>
        </div>
      )}

      {/* ── 3. Live Active Artifacts ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#8b949e',
          textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Active Artifacts</div>
        {activeInfos.length === 0 ? (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: 'rgba(63,185,80,0.06)', border: '1px solid rgba(63,185,80,0.2)',
            borderRadius: 10, padding: '14px 18px',
            fontSize: 13, color: '#3fb950', fontWeight: 600,
          }}>
            <span>✓</span>
            <span>No artifacts detected this frame — signal is clean</span>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {activeInfos.map(info => (
              <ActiveArtifactCard key={info.id} info={info} isLive={activeSet.has(info.id)} />
            ))}
          </div>
        )}
      </div>

      {/* ── 4. Recommended Pipeline Checklist ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#8b949e',
          textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
          Pre-Session Pipeline Checklist
        </div>
        <div style={{
          background: '#161b22', border: '1px solid #30363d',
          borderRadius: 10, padding: '14px 18px',
        }}>
          <PipelineChecklist />
        </div>
      </div>

      {/* ── 5. Consumer-Grade Caveats ── */}
      <ConsumerCaveats />

      {/* ── 6. Encyclopedia ── */}
      <div>
        <div
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            cursor: 'pointer', padding: '10px 14px',
            background: '#161b22', border: '1px solid #30363d', borderRadius: 10,
          }}
          onClick={() => setEncyclopediaOpen(o => !o)}
          role="button"
          aria-expanded={encyclopediaOpen}
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && setEncyclopediaOpen(o => !o)}
        >
          <span style={{ fontSize: 13, fontWeight: 700, color: '#cdd9e5' }}>📖 EEG Artifact Encyclopedia</span>
          <span style={{ fontSize: 12, color: '#8b949e', transition: 'transform 0.2s',
            transform: encyclopediaOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
        </div>
        {encyclopediaOpen && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
            <div style={{ fontSize: 12, color: '#8b949e', padding: '4px 2px' }}>
              Complete reference for all 7 EEG artifact types — identification, immediate actions,
              prevention, and pipeline stage notes. Expand each entry to read.
            </div>
            {ARTIFACT_ENCYCLOPEDIA.map(info => (
              <EncyclopediaEntry key={info.id} info={info} />
            ))}
          </div>
        )}
      </div>

    </div>
  )
}
