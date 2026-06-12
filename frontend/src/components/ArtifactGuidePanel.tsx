/**
 * ArtifactGuidePanel
 *
 * Live artifact intelligence panel for Neurolink-v1.
 *
 * What it does:
 *   1. Reads NeurolinkState fields (artifact_rejected, artifact_reasons,
 *      motion_rms, bad_channels, eeg_samples) to classify which artifact
 *      type is currently present.
 *   2. Shows a real-time signal-quality score (0–100) derived from the
 *      rejection rate, bad-channel count, and motion level.
 *   3. Displays type-specific identification cues and remediation steps
 *      for every active artifact class.
 *   4. Provides a collapsible EEG Artifact Encyclopedia with detailed
 *      information on all 7 artifact types: Ocular, EMG/Muscle,
 *      Cardiac/BCG, Movement, Power-line, Electrode Pop/Drift,
 *      and Bad Channel.
 *
 * Usage:
 *   <ArtifactGuidePanel state={neurolinkState} rejectRate={0.12} />
 *
 * Designed to sit inside the same Filters/Artifacts tab as
 * ArtifactConfigPanel and ArtifactStatsPanel.
 */
import React, { useState, useEffect, useRef } from 'react'
import type { NeurolinkState } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ArtifactGuidePanelProps {
  /** Full NeurolinkState from useNeurolinkStream or similar. */
  state: NeurolinkState | null
  /** Rolling rejection rate (0–1) from useArtifactStats. */
  rejectRate: number
  /** Whether a device is currently connected. */
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
// Artifact Encyclopedia Data
// ─────────────────────────────────────────────────────────────────────────────

const ARTIFACT_ENCYCLOPEDIA: ArtifactInfo[] = [
  {
    id: 'ocular',
    label: 'Ocular (Eye) Artifact',
    icon: '👁',
    colour: '#58a6ff',
    borderColour: 'rgba(88,166,255,0.35)',
    bgColour: 'rgba(88,166,255,0.07)',
    frequencyRange: '0.1 – 10 Hz (overlaps delta & theta)',
    affectedChannels: 'Fp1, Fp2, AF3, AF4 (frontal)',
    waveformCue: 'Slow, high-amplitude corneoretinal potentials; blink peaks are sharp with rounded recovery',
    identificationCues: [
      'Sudden large-amplitude spike on frontal channels (Fp1/Fp2) lasting ~200–400 ms — typical blink',
      'Slow lateral drift on F7/F8 during horizontal saccades',
      'Amplitude often 100–500 µV — far exceeding typical EEG (10–50 µV)',
      'Waveform does NOT follow 10/20 scalp topography — disproportionately frontal',
      'Correlates with visible blinking or eye movement in the subject',
    ],
    immediateActions: [
      'Instruct the subject to keep eyes soft-focused or closed during critical recording windows',
      'The Amplitude Gate (Stage 3) will automatically reject these epochs if pk2pk_uv ≤ 150 µV',
      'For meditation sessions with closed eyes, blink artifacts are minimal — open eyes practice needs wider gating',
      'If using an EOG reference channel, enable the Gratton–Coles regression in the backend pipeline',
    ],
    preventionTips: [
      'Closed-eye meditation dramatically reduces blink artifact prevalence',
      'Brief pre-session eye exercises to fatigue the blink reflex can help during open-eye protocols',
      'Electrode gel quality on Fp1/Fp2 matters most — poor contact amplifies ocular spread',
    ],
    signalStageNote: 'Caught by Stage 3 Amplitude Gate if threshold ≤ 150 µV. ICA-based removal is available post-hoc in MNE-Python via the ICLabel classifier.',
  },
  {
    id: 'muscle',
    label: 'EMG / Muscle Artifact',
    icon: '💪',
    colour: '#d27bff',
    borderColour: 'rgba(210,123,255,0.35)',
    bgColour: 'rgba(210,123,255,0.07)',
    frequencyRange: '20 – 300 Hz (broadband high-frequency)',
    affectedChannels: 'Temporal (T3/T4/T5/T6), frontal — wherever muscles underlie electrodes',
    waveformCue: 'Dense, irregular, high-frequency bursts with no repeating structure — looks like "grass" in the raw trace',
    identificationCues: [
      'Sudden broadband power increase above 20 Hz — especially obvious in the 40–80 Hz gamma band',
      'Distribution follows jaw, temple, or neck muscle anatomy — not neural topography',
      'Kurtosis spikes above 5 (leptokurtic distribution) — the Kurtosis Gate catches this',
      'Millisecond-scale spikes cluster into short bursts (50–200 ms) then disappear',
      'Jaw clenching produces bilateral temporal contamination; neck tension affects posterior channels',
    ],
    immediateActions: [
      'Ask the subject to relax the jaw — slightly open mouth, tongue resting on lower palate',
      'Neck and shoulder rolling for 30 seconds before session start prevents anticipatory tension',
      'The Kurtosis Burst Gate (Stage 3) at threshold 5.0 k rejects the most severe bursts automatically',
      'Lower the Kurtosis threshold to 3.5 k if gamma contamination is suspected during analysis',
    ],
    preventionTips: [
      'Pre-session progressive muscle relaxation from feet to scalp is the single best prevention',
      'Avoid caffeinated beverages 2 hours before recording — caffeine increases jaw tension',
      'Ensure the electrode headset is not too tight — mechanical pressure on temporal muscles causes involuntary tension',
    ],
    signalStageNote: 'Caught by Stage 3 Kurtosis Gate. Residual muscle noise above 40 Hz can be attenuated with a low-pass filter cutoff in FiltersPage. Gamma band analysis should always note potential EMG contamination.',
  },
  {
    id: 'cardiac',
    label: 'Cardiac / BCG Artifact',
    icon: '❤️',
    colour: '#f85149',
    borderColour: 'rgba(248,81,73,0.35)',
    bgColour: 'rgba(248,81,73,0.07)',
    frequencyRange: '~1.2 Hz fundamental + harmonics (1–10 Hz)',
    affectedChannels: 'Temporal arteries (T3/T4), vertex (Cz) in upright subjects',
    waveformCue: 'Rhythmic sharp deflection occurring once per heartbeat (~0.8–1.2 s period); waveform resembles a QRS complex',
    identificationCues: [
      'Periodic, rhythmic artifact with consistent period matching the heart rate (typically 0.8–1.2 s)',
      'Amplitude usually 5–30 µV — smaller than ocular artifacts but consistent',
      'Visible on HRV panel as regular pulse — compare artifact timing to hr_bpm field',
      'Waveform shape is stereotyped and reproducible across cycles',
      'Increases in amplitude with elevated heart rate (after exercise or stress)',
    ],
    immediateActions: [
      'If HR BPM display is active, compare artifact periodicity to hr_bpm — confirmation of cardiac origin',
      'Ensure electrode impedance is low on temporal channels — high impedance amplifies cardiac coupling',
      'Average re-referencing (Common Average Reference) partially suppresses this artifact',
      'ICA with ICLabel labels this component "Heart" — remove post-hoc if offline analysis is needed',
    ],
    preventionTips: [
      'Record at least 2–3 minutes of resting baseline before meditation onset — allows heart rate to stabilize',
      'Upright seated position with relaxed shoulders minimises carotid pulse coupling to electrodes',
      'Avoid recording immediately after cardiovascular exercise',
    ],
    signalStageNote: 'Not directly caught by current Stage 3 gates (it falls below the amplitude threshold). Post-hoc ICA removal in MNE-Python is the standard method. A future enhancement could add an ECG reference channel for online regression.',
  },
  {
    id: 'movement',
    label: 'Movement / Motion Artifact',
    icon: '🏃',
    colour: '#e3b341',
    borderColour: 'rgba(227,179,65,0.35)',
    bgColour: 'rgba(227,179,65,0.07)',
    frequencyRange: '0.1 – 5 Hz (low-frequency drift and transients)',
    affectedChannels: 'All channels (global) for head movement; localized for cable sway',
    waveformCue: 'Slow-wave drift with occasional sharp transients when movement begins or ends; amplitude can be extreme (>500 µV)',
    identificationCues: [
      'IMU motion_rms field exceeds 0.15 g — direct physical evidence of movement',
      'All channels affected simultaneously — distinguishes it from electrode-specific noise',
      'Low-frequency (<3 Hz) large-amplitude waves that are aperiodic',
      'Abrupt onset and offset matching body movement events',
      'Cable-drag artifacts appear on individual channels as sharp asymmetric spikes',
    ],
    immediateActions: [
      'The IMU Motion Gate (Stage 3) at 0.15 g automatically rejects these frames — verify it is enabled',
      'Check motion_rms on the IMU Panel — values >0.15 g confirm movement contamination',
      'For walking meditation: raise the IMU gate threshold to 0.30 g to retain usable data',
      'Ensure electrode cable routes are not slack — cable movement is a major source of localized motion artifacts',
    ],
    preventionTips: [
      'Secure all cables along the headband or use wireless streaming exclusively',
      'Seated or supine recording positions with minimal body movement are ideal',
      'For mobile meditation protocols, collect a clean resting calibration segment first — ASR uses it as reference',
    ],
    signalStageNote: 'Caught by Stage 3 IMU Motion Gate. For prolonged sessions, the IMU threshold may need session-specific tuning. The motion_rms value is streamed live in NeurolinkState.',
  },
  {
    id: 'powerline',
    label: 'Power-Line Interference',
    icon: '⚡',
    colour: '#3fb950',
    borderColour: 'rgba(63,185,80,0.35)',
    bgColour: 'rgba(63,185,80,0.07)',
    frequencyRange: '50 Hz (Europe/Asia) or 60 Hz (Americas) — sharp tonal',
    affectedChannels: 'All channels (global, equal amplitude)',
    waveformCue: 'Perfectly sinusoidal, constant-amplitude oscillation at exactly 50 or 60 Hz — visible as a bright horizontal stripe in the spectrogram',
    identificationCues: [
      'PSD shows a sharp, narrow spike at exactly 50 Hz or 60 Hz — much narrower than broadband EMG',
      'Constant amplitude and frequency — does not fluctuate with cognitive state',
      'Visible as a horizontal band in the RollingSpectrogram at the line frequency',
      'Amplitude increases near poorly shielded power equipment (monitors, fluorescent lights, laptop chargers)',
      'Unaffected by subject behaviour — it does not track blinking or muscle activity',
    ],
    immediateActions: [
      'Verify that the Notch Filter (50/60 Hz) is enabled in FiltersPage — this is the primary defense',
      'Move the recording setup away from switching power supplies, USB hubs, and fluorescent lights',
      'If using a laptop, switch to battery power during critical recording windows',
      'Ensure electrode cables are not routed parallel to power cables',
    ],
    preventionTips: [
      'Use a shielded electrode cable system or a Faraday cage environment for research-grade recordings',
      'Battery-powered amplifiers avoid ground loops that couple mains noise',
      'Keep the recording distance from the router/hub to the EEG amplifier as short as possible',
    ],
    signalStageNote: 'Handled in Stage 1 by the Notch Filter in FiltersPage. The filter frequency is configurable for 50 Hz (EU) or 60 Hz (US/Canada). A correctly applied notch filter leaves all other frequencies perfectly intact.',
  },
  {
    id: 'electrode',
    label: 'Electrode Pop / Impedance Drift',
    icon: '🔌',
    colour: '#ffa657',
    borderColour: 'rgba(255,166,87,0.35)',
    bgColour: 'rgba(255,166,87,0.07)',
    frequencyRange: '< 0.5 Hz (drift) + wideband transient (pop)',
    affectedChannels: 'Single affected channel — key diagnostic feature',
    waveformCue: 'A single channel shows an abrupt step-change ("pop") or slow monotonic drift away from baseline while all other channels are clean',
    identificationCues: [
      'Affects exactly ONE channel while all others remain normal — critical localisation clue',
      'Electrode pop: sudden large-amplitude step (can be >1000 µV) followed by slow return to baseline',
      'Impedance drift: slow wandering away from zero (DC offset) increasing over minutes',
      'Check ImpedancePanel — kΩ values >50 kΩ indicate compromised contact',
      'Sweat accumulation under dry electrodes causes gradual impedance reduction but also chemical drift artifacts',
    ],
    immediateActions: [
      'Check ImpedancePanel for the affected channel — re-seat the electrode if impedance is >50 kΩ',
      'If using gel electrodes, re-apply conductive gel to the affected electrode',
      'For dry electrodes, gentle pressure and small circular motion can restore contact temporarily',
      'The Bad Channel detection (Stage 2) will flag and remove the affected channel automatically',
    ],
    preventionTips: [
      'Before every session, verify all channel impedances are <20 kΩ (green in ImpedancePanel)',
      'For dry electrode systems, ensure scalp is clean — oils and hair products increase contact resistance',
      'In long sessions (>30 min), plan a brief mid-session impedance check to catch gel drying',
    ],
    signalStageNote: 'The Bad Channel detection pipeline (Stage 2, BadChannelPanel) identifies and interpolates affected channels. The High-Pass Filter (0.5 Hz) in Stage 1 attenuates slow drift. Electrode pops are caught by the Amplitude Gate in Stage 3.',
  },
  {
    id: 'badchannel',
    label: 'Bad Channel (Flat / Noisy)',
    icon: '📡',
    colour: '#8b949e',
    borderColour: 'rgba(139,148,158,0.35)',
    bgColour: 'rgba(139,148,158,0.07)',
    frequencyRange: 'Flat: DC (zero signal) | Noisy: wideband',
    affectedChannels: 'Any channel — identified by abnormal PSD relative to neighbours',
    waveformCue: 'Flat channel: constant zero or near-zero signal. Noisy channel: much higher variance than all other channels with no coherent pattern',
    identificationCues: [
      'BadChannelPanel highlights the channel by name — this is the authoritative real-time source',
      'Flat channel: PSD is orders of magnitude below all other channels across all frequencies',
      'Noisy channel: PSD is uniformly elevated 2–3× above neighbours — no spectral peaks',
      'Spatial distribution check: if one channel reads very differently from its neighbours, it is likely bad',
      'Contact quality indicator shows red or yellow for the affected electrode',
    ],
    immediateActions: [
      'Check bad_channels field in NeurolinkState — automatically flagged by Stage 2 pipeline',
      'Bad channels are excluded from band-power computation and EA1 scoring automatically',
      'Re-seat the electrode before the next epoch to attempt recovery',
      'If persistent: record the channel name in your session notes as excluded from analysis',
    ],
    preventionTips: [
      'Run a 30-second impedance check before every session — pre-empts bad channel formation',
      'For gel systems, apply enough gel to fill the electrode cup without shorting adjacent channels',
      'Dry electrode systems require clean, hair-free scalp contact — part hair at electrode sites if needed',
    ],
    signalStageNote: 'Handled by Stage 2 bad channel detection. The bad_channels string array in NeurolinkState lists affected channels by name each frame. Spherical spline interpolation is applied in the backend when channel count permits.',
  },
]

// ─────────────────────────────────────────────────────────────────────────────
// Mapping: artifact_reasons strings → ArtifactClass
// ─────────────────────────────────────────────────────────────────────────────

const REASON_TO_CLASS: Record<string, ArtifactClass> = {
  amplitude: 'ocular',   // amplitude spikes are most commonly ocular in frontal EEG
  kurtosis:  'muscle',   // kurtosis bursts are the EMG signature
  motion:    'movement', // IMU gate
  imu:       'movement',
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
  // Start at 100 and subtract penalty points
  let score = 100
  // Rejection rate penalty: up to -40 pts at 100% rejection
  score -= rejectRate * 40
  // Bad channels penalty: -10 pts per bad channel, max -30
  score -= Math.min(badChannelCount * 10, 30)
  // Motion penalty: >0.15 g is already gated; penalise sub-threshold motion
  if (motionRms !== null && motionRms > 0.05) {
    score -= Math.min((motionRms - 0.05) * 40, 15)
  }
  // Poor contact flag: -15 pts
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
// Determine active artifact classes from current frame
// ─────────────────────────────────────────────────────────────────────────────

function detectActiveArtifacts(
  state: NeurolinkState | null,
): ArtifactClass[] {
  if (!state) return []
  const active = new Set<ArtifactClass>()

  // Stage 3 rejection reasons
  if (state.artifact_rejected && state.artifact_reasons) {
    for (const reason of state.artifact_reasons) {
      const cls = REASON_TO_CLASS[reason.toLowerCase()]
      if (cls) active.add(cls)
    }
  }

  // Bad channels → electrode or bad channel artifact
  if (state.bad_channels && state.bad_channels.length > 0) {
    active.add('badchannel')
  }

  // Poor contact flag
  if (state.poor_contact) {
    active.add('electrode')
  }

  return Array.from(active)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

const S: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', gap: 18 },
  sectionTitle: {
    fontSize: 13, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6,
  },
  qualityRow: {
    display: 'flex', alignItems: 'center', gap: 16,
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 10, padding: '14px 18px',
  },
  qualityScore: { fontSize: 36, fontWeight: 800, lineHeight: 1 },
  qualityMeta: { flex: 1 },
  qualityLabel: { fontSize: 14, fontWeight: 700, marginBottom: 4 },
  qualityDesc: { fontSize: 12, color: '#8b949e', lineHeight: 1.5 },
  barBg: {
    height: 6, background: '#21262d', borderRadius: 3,
    overflow: 'hidden', marginTop: 6, width: '100%',
  },
  activeSection: { display: 'flex', flexDirection: 'column', gap: 10 },
  activeCard: {
    borderRadius: 10, padding: '14px 18px',
    display: 'flex', flexDirection: 'column', gap: 10,
  },
  activeHeader: {
    display: 'flex', alignItems: 'center', gap: 10,
  },
  activeIcon: { fontSize: 22, lineHeight: 1 },
  activeName: { fontSize: 14, fontWeight: 700, color: '#cdd9e5', flex: 1 },
  activeLive: {
    fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
    padding: '2px 8px', borderRadius: 10,
    background: 'rgba(248,81,73,0.15)', border: '1px solid rgba(248,81,73,0.4)',
    color: '#f85149',
  },
  activeTagRow: { display: 'flex', gap: 6, flexWrap: 'wrap' as const },
  tag: {
    fontSize: 11, fontWeight: 600, padding: '2px 8px',
    borderRadius: 6, background: 'rgba(139,148,158,0.1)',
    border: '1px solid rgba(139,148,158,0.2)', color: '#8b949e',
  },
  cuesList: { display: 'flex', flexDirection: 'column', gap: 5 },
  cue: {
    fontSize: 12, color: '#cdd9e5', lineHeight: 1.5,
    paddingLeft: 14, position: 'relative' as const,
  },
  actionsList: { display: 'flex', flexDirection: 'column', gap: 5 },
  action: {
    fontSize: 12, color: '#3fb950', lineHeight: 1.5,
    paddingLeft: 14, position: 'relative' as const,
  },
  subLabel: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase', letterSpacing: 0.7, marginBottom: 3,
  },
  stageNote: {
    fontSize: 11, color: '#484f58', background: '#0d1117',
    border: '1px solid #21262d', borderRadius: 6, padding: '6px 10px',
    lineHeight: 1.5,
  },
  encyclopediaToggle: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    cursor: 'pointer', padding: '10px 14px',
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 10,
    transition: 'border-color 0.15s',
  },
  encyclopediaTitle: { fontSize: 13, fontWeight: 700, color: '#cdd9e5' },
  encyclopediaChevron: { fontSize: 12, color: '#8b949e', transition: 'transform 0.2s' },
  encyclopediaBody: {
    display: 'flex', flexDirection: 'column', gap: 8,
    marginTop: 4,
  },
  encEntry: {
    borderRadius: 8, border: '1px solid #21262d',
    overflow: 'hidden',
  },
  encEntryHeader: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 14px', cursor: 'pointer',
    background: '#161b22',
    transition: 'background 0.15s',
  },
  encEntryIcon: { fontSize: 16 },
  encEntryLabel: { fontSize: 13, fontWeight: 600, color: '#cdd9e5', flex: 1 },
  encEntryChevron: { fontSize: 11, color: '#8b949e', transition: 'transform 0.2s' },
  encEntryBody: {
    padding: '12px 14px',
    background: '#0d1117',
    display: 'flex', flexDirection: 'column', gap: 10,
  },
  encRow: { display: 'flex', flexDirection: 'column', gap: 3 },
  encRowLabel: {
    fontSize: 10, fontWeight: 700, color: '#484f58',
    textTransform: 'uppercase', letterSpacing: 0.7,
  },
  encRowValue: { fontSize: 12, color: '#8b949e', lineHeight: 1.5 },
  encCueItem: {
    fontSize: 12, color: '#cdd9e5', lineHeight: 1.5,
    paddingLeft: 12, position: 'relative' as const,
  },
  encActionItem: {
    fontSize: 12, color: '#3fb950', lineHeight: 1.5,
    paddingLeft: 12, position: 'relative' as const,
  },
  encPreventItem: {
    fontSize: 12, color: '#58a6ff', lineHeight: 1.5,
    paddingLeft: 12, position: 'relative' as const,
  },
  noArtifacts: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'rgba(63,185,80,0.06)', border: '1px solid rgba(63,185,80,0.2)',
    borderRadius: 10, padding: '14px 18px',
    fontSize: 13, color: '#3fb950', fontWeight: 600,
  },
  disconnected: {
    color: '#484f58', fontSize: 13, padding: '8px 0',
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// EncyclopediaEntry — single collapsible artifact detail
// ─────────────────────────────────────────────────────────────────────────────

function EncyclopediaEntry({ info }: { info: ArtifactInfo }) {
  const [open, setOpen] = useState(false)

  return (
    <div style={S.encEntry}>
      <div
        style={S.encEntryHeader}
        onClick={() => setOpen(o => !o)}
        role="button"
        aria-expanded={open}
        tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setOpen(o => !o)}
      >
        <span style={S.encEntryIcon}>{info.icon}</span>
        <span style={S.encEntryLabel}>{info.label}</span>
        <span style={{ ...S.encEntryChevron, transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
      </div>

      {open && (
        <div style={S.encEntryBody}>
          <div style={S.encRow}>
            <div style={S.encRowLabel}>Frequency Range</div>
            <div style={S.encRowValue}>{info.frequencyRange}</div>
          </div>
          <div style={S.encRow}>
            <div style={S.encRowLabel}>Affected Channels</div>
            <div style={S.encRowValue}>{info.affectedChannels}</div>
          </div>
          <div style={S.encRow}>
            <div style={S.encRowLabel}>Waveform Signature</div>
            <div style={S.encRowValue}>{info.waveformCue}</div>
          </div>

          <div style={S.encRow}>
            <div style={S.encRowLabel}>How to Identify</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 2 }}>
              {info.identificationCues.map((c, i) => (
                <div key={i} style={S.encCueItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#484f58' }}>·</span>
                  {c}
                </div>
              ))}
            </div>
          </div>

          <div style={S.encRow}>
            <div style={S.encRowLabel}>Immediate Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 2 }}>
              {info.immediateActions.map((a, i) => (
                <div key={i} style={S.encActionItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#2ea043' }}>→</span>
                  {a}
                </div>
              ))}
            </div>
          </div>

          <div style={S.encRow}>
            <div style={S.encRowLabel}>Prevention</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 2 }}>
              {info.preventionTips.map((p, i) => (
                <div key={i} style={S.encPreventItem}>
                  <span style={{ position: 'absolute', left: 0, color: '#388bfd' }}>◈</span>
                  {p}
                </div>
              ))}
            </div>
          </div>

          <div style={{ ...S.stageNote, marginTop: 2 }}>
            <span style={{ color: '#388bfd', fontWeight: 700 }}>Pipeline note: </span>
            {info.signalStageNote}
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ActiveArtifactCard — card shown when an artifact is live
// ─────────────────────────────────────────────────────────────────────────────

function ActiveArtifactCard({ info }: { info: ArtifactInfo }) {
  return (
    <div style={{
      ...S.activeCard,
      background: info.bgColour,
      border: `1px solid ${info.borderColour}`,
    }}>
      <div style={S.activeHeader}>
        <span style={S.activeIcon}>{info.icon}</span>
        <span style={S.activeName}>{info.label}</span>
        <span style={S.activeLive}>LIVE</span>
      </div>

      <div style={S.activeTagRow}>
        <span style={S.tag}>⌖ {info.frequencyRange}</span>
        <span style={S.tag}>⊙ {info.affectedChannels}</span>
      </div>

      <div>
        <div style={S.subLabel}>How to identify this</div>
        <div style={S.cuesList}>
          {info.identificationCues.slice(0, 3).map((c, i) => (
            <div key={i} style={S.cue}>
              <span style={{ position: 'absolute', left: 0, color: '#484f58' }}>·</span>
              {c}
            </div>
          ))}
        </div>
      </div>

      <div>
        <div style={S.subLabel}>Immediate actions</div>
        <div style={S.actionsList}>
          {info.immediateActions.slice(0, 3).map((a, i) => (
            <div key={i} style={S.action}>
              <span style={{ position: 'absolute', left: 0, color: '#2ea043' }}>→</span>
              {a}
            </div>
          ))}
        </div>
      </div>

      <div style={S.stageNote}>
        <span style={{ color: '#388bfd', fontWeight: 700 }}>Pipeline: </span>
        {info.signalStageNote}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal Quality Gauge
// ─────────────────────────────────────────────────────────────────────────────

function SignalQualityGauge({
  state, rejectRate,
}: {
  state: NeurolinkState | null
  rejectRate: number
}) {
  const badCount = state?.bad_channels?.length ?? 0
  const motionRms = state?.motion_rms ?? null
  const poorContact = state?.poor_contact ?? false
  const score = computeSignalQuality(rejectRate, badCount, motionRms, poorContact)
  const { label, colour } = qualityLabel(score)

  // Build contextual description
  const issues: string[] = []
  if (rejectRate >= 0.10) issues.push(`${(rejectRate * 100).toFixed(0)}% frames rejected`)
  if (badCount > 0)       issues.push(`${badCount} bad channel${badCount > 1 ? 's' : ''}`)
  if (poorContact)        issues.push('poor electrode contact')
  if (motionRms !== null && motionRms > 0.15) issues.push(`motion ${motionRms.toFixed(2)} g`)

  const desc = issues.length > 0
    ? `Issues: ${issues.join(' · ')}`
    : 'Signal is clean — no active degradation detected'

  return (
    <div style={S.qualityRow}>
      <div style={{ ...S.qualityScore, color: colour }}>{score}</div>
      <div style={S.qualityMeta}>
        <div style={{ ...S.qualityLabel, color: colour }}>Signal Quality · {label}</div>
        <div style={S.qualityDesc}>{desc}</div>
        <div style={S.barBg}>
          <div style={{
            height: '100%',
            width: `${score}%`,
            background: colour,
            borderRadius: 3,
            transition: 'width 0.4s ease, background 0.4s ease',
          }} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// useRecentActiveClasses — keep artifact cards visible for 3 s after clearing
// ─────────────────────────────────────────────────────────────────────────────

function useRecentActiveClasses(
  active: ArtifactClass[],
  holdMs: number,
): ArtifactClass[] {
  const [displayed, setDisplayed] = useState<ArtifactClass[]>([])
  const timersRef = useRef<Map<ArtifactClass, ReturnType<typeof setTimeout>>>(new Map())

  useEffect(() => {
    const activeSet = new Set(active)

    // Add new active classes immediately
    setDisplayed(prev => {
      const next = new Set(prev)
      for (const cls of active) next.add(cls)
      return Array.from(next)
    })

    // Schedule removal for classes that just cleared
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
          // Class is active again — cancel pending removal
          const t = timersRef.current.get(cls)
          if (t !== undefined) {
            clearTimeout(t)
            timersRef.current.delete(cls)
          }
        }
      }
      return prev
    })
  }, [active, holdMs])

  return displayed
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

  const activeClasses = detectActiveArtifacts(state)
  const displayedClasses = useRecentActiveClasses(activeClasses, 3000)

  const activeInfos = displayedClasses
    .map(cls => ARTIFACT_ENCYCLOPEDIA.find(e => e.id === cls))
    .filter((e): e is ArtifactInfo => e !== undefined)

  if (!connected) {
    return <div style={S.disconnected}>Connect a device to enable live artifact guidance.</div>
  }

  return (
    <div style={S.root}>
      {/* ── Signal Quality Gauge ── */}
      <div>
        <div style={S.sectionTitle}>Signal Quality</div>
        <SignalQualityGauge state={state} rejectRate={rejectRate} />
      </div>

      {/* ── Live Active Artifacts ── */}
      <div>
        <div style={S.sectionTitle}>Active Artifacts</div>
        {activeInfos.length === 0 ? (
          <div style={S.noArtifacts}>
            <span>✓</span>
            <span>No artifacts detected this frame — signal is clean</span>
          </div>
        ) : (
          <div style={S.activeSection}>
            {activeInfos.map(info => (
              <ActiveArtifactCard key={info.id} info={info} />
            ))}
          </div>
        )}
      </div>

      {/* ── Encyclopedia ── */}
      <div>
        <div
          style={S.encyclopediaToggle}
          onClick={() => setEncyclopediaOpen(o => !o)}
          role="button"
          aria-expanded={encyclopediaOpen}
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && setEncyclopediaOpen(o => !o)}
        >
          <span style={S.encyclopediaTitle}>📖 EEG Artifact Encyclopedia</span>
          <span style={{
            ...S.encyclopediaChevron,
            transform: encyclopediaOpen ? 'rotate(180deg)' : 'rotate(0deg)',
          }}>▾</span>
        </div>

        {encyclopediaOpen && (
          <div style={S.encyclopediaBody}>
            <div style={{ fontSize: 12, color: '#8b949e', padding: '4px 2px' }}>
              Complete reference for all EEG artifact types — identification,
              immediate actions, prevention, and pipeline stage notes.
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
