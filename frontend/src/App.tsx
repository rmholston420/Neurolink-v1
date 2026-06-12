import React, { useState } from 'react'
import { useNeurolinkSSE } from './hooks/useNeurolinkSSE'
import { useMuseBLE } from './hooks/useMuseBLE'
import { useArtifactStats } from './hooks/useArtifactStats'

// Core components
import BandPowerChart    from './components/BandPowerChart'
import SSpaceDisplay    from './components/SSpaceDisplay'
import EA1Score         from './components/EA1Score'
import HRVPanel         from './components/HRVPanel'
import ContactQuality   from './components/ContactQuality'
import BreathingPanel   from './components/BreathingPanel'
import IMUPanel         from './components/IMUPanel'
import CalibrationPanel from './components/CalibrationPanel'
import ConnectionPanel  from './components/ConnectionPanel'
import DeviceStatusBar  from './components/DeviceStatusBar'

// Stage 0-3 pipeline visualisation components
import BadChannelPanel     from './components/BadChannelPanel'
import ArtifactStatsPanel  from './components/ArtifactStatsPanel'
import SignalPipelinePanel from './components/SignalPipelinePanel'
import ImpedancePanel      from './components/ImpedancePanel'

// Stage 3 config panel
import ArtifactConfigPanel from './components/ArtifactConfigPanel'

// Artifact intelligence + encyclopedia
import ArtifactGuidePanel from './components/ArtifactGuidePanel'

// Visualisation components
import RollingSpectrogram  from './components/RollingSpectrogram'
import TopoMap             from './components/TopoMap'
import BandTrend           from './components/BandTrend'
import ConnectivityArc     from './components/ConnectivityArc'
import NeurofeedbackGauge  from './components/NeurofeedbackGauge'

// ── Tier 1 feature components & hooks ────────────────────────────────────────
import AudioFeedbackPanel  from './components/AudioFeedbackPanel'
import WanderingLog        from './components/WanderingLog'
import SessionHistoryPanel from './components/SessionHistoryPanel'
import { useAudioFeedback }     from './hooks/useAudioFeedback'
import { useWanderingDetector } from './hooks/useWanderingDetector'
import { useSessionHistory }    from './hooks/useSessionHistory'

// ── Tier 2 feature components & hooks ────────────────────────────────────────
import MeditationTimer      from './components/MeditationTimer'
import PersonalBaseline     from './components/PersonalBaseline'
import HRVCoherenceTrainer  from './components/HRVCoherenceTrainer'
import AlchemicalJournal    from './components/AlchemicalJournal'
import SessionGoals         from './components/SessionGoals'
import { usePersonalBaseline }  from './hooks/usePersonalBaseline'
import { useHRVCoherence }      from './hooks/useHRVCoherence'
import { useAlchemicalJournal } from './hooks/useAlchemicalJournal'
import { useSessionGoals }      from './hooks/useSessionGoals'

// ── Filters page ─────────────────────────────────────────────────────────────
import FiltersPage from './components/FiltersPage'

const API_URL = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000'

// ── Styles ────────────────────────────────────────────────────────────────────
const S: Record<string, React.CSSProperties> = {
  app: {
    maxWidth: 1280,
    margin: '0 auto',
    padding: '20px 16px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: '#cdd9e5',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 20, paddingBottom: 14, borderBottom: '1px solid #30363d',
    gap: 12, flexWrap: 'wrap' as const,
  },
  headerLeft: {
    display: 'flex', alignItems: 'center', gap: 14,
  },
  title: { fontSize: 22, fontWeight: 700, color: '#58a6ff', letterSpacing: '-0.5px' },
  tabBar: {
    display: 'flex', gap: 2, marginBottom: 18,
    borderBottom: '1px solid #30363d', paddingBottom: 0,
    flexWrap: 'wrap' as const,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: 14,
  },
  gridWide: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
    gap: 14,
  },
  card: {
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 10, padding: 18,
  },
  cardWide: {
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 10, padding: 18, gridColumn: '1 / -1',
  },
  cardTitle: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase' as const, letterSpacing: 1, marginBottom: 12,
  },
  frameCount: {
    fontSize: 11, color: '#484f58', marginTop: 14, textAlign: 'center' as const,
  },
  // Divider between FiltersPage sections
  divider: {
    display: 'flex', alignItems: 'center', gap: 12,
    margin: '8px 0',
  },
  dividerLine: {
    flex: 1, height: 1, background: '#21262d',
  },
  dividerLabel: {
    fontSize: 10, fontWeight: 700, color: '#484f58',
    textTransform: 'uppercase' as const, letterSpacing: 1, flexShrink: 0,
  },
}

const statusBadge = (connected: boolean): React.CSSProperties => ({
  display: 'inline-flex', alignItems: 'center', gap: 6,
  padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
  background: connected ? 'rgba(46,160,67,0.12)' : 'rgba(248,81,73,0.12)',
  color: connected ? '#3fb950' : '#f85149',
  border: `1px solid ${connected ? '#238636' : '#da3633'}`,
})

const dot = (connected: boolean): React.CSSProperties => ({
  width: 7, height: 7, borderRadius: '50%',
  background: connected ? '#3fb950' : '#f85149',
})

type Tab = 'live' | 'pipeline' | 'spectrogram' | 'topo' | 'connectivity' | 'practice' | 'journal' | 'history' | 'filters'
const TABS: { id: Tab; label: string }[] = [
  { id: 'live',          label: '⚡ Live' },
  { id: 'pipeline',      label: '🔬 Pipeline' },
  { id: 'spectrogram',   label: '🌊 Spectrogram' },
  { id: 'topo',          label: '🧠 Topo Map' },
  { id: 'connectivity',  label: '🔗 Connectivity' },
  { id: 'practice',      label: '🧘 Practice' },
  { id: 'journal',       label: '📓 Journal' },
  { id: 'history',       label: '📊 History' },
  { id: 'filters',       label: '🔧 Filters' },
]

function tabBtn(id: Tab, active: Tab): React.CSSProperties {
  const isActive = id === active
  return {
    padding: '7px 16px', fontSize: 13, fontWeight: 600,
    border: 'none', cursor: 'pointer', background: 'none',
    color: isActive ? '#58a6ff' : '#8b949e',
    borderBottom: `2px solid ${isActive ? '#388bfd' : 'transparent'}`,
    marginBottom: -1,
    transition: 'color 0.15s, border-color 0.15s',
  }
}

function syntheticChannelBands(
  bands: { alpha: number; theta: number; beta: number; delta: number; gamma: number } | null,
  band: 'alpha' | 'theta' | 'beta' | 'delta' | 'gamma'
): number[] | null {
  if (!bands) return null
  const base = bands[band]
  return [0.9, 1.1, 1.05, 0.95].map(f => base * f)
}

function syntheticEEGSamples(
  bands: { alpha: number } | null
): number[][] | null {
  if (!bands) return null
  const base = bands.alpha
  return Array.from({ length: 4 }, (_, ch) => {
    const freq = 8 + ch * 1.2
    return Array.from({ length: 128 }, (__, n) =>
      base * 200 * Math.sin(2 * Math.PI * freq * n / 256) +
      (Math.random() - 0.5) * base * 50
    )
  })
}

function bleContactToQuality(contact: { tp9: boolean; af7: boolean; af8: boolean; tp10: boolean }): number {
  const goods = [contact.tp9, contact.af7, contact.af8, contact.tp10].filter(Boolean).length
  return goods / 4
}

export default function App() {
  const state      = useNeurolinkSSE(`${API_URL}/api/v1/neurolink/stream`)
  const ble        = useMuseBLE(API_URL)
  const connected  = state?.connected ?? false
  const [tab, setTab] = useState<Tab>('live')

  // ── Stage 0-3 stats hook ──────────────────────────────────────────────────
  const artifactStats = useArtifactStats(state)

  // ── Tier 1 hooks ──────────────────────────────────────────────────────────
  const audio    = useAudioFeedback(state)
  const detector = useWanderingDetector(state)
  const history  = useSessionHistory(state)

  // ── Tier 2 hooks ──────────────────────────────────────────────────────────
  const baseline    = usePersonalBaseline(state)
  const coherence   = useHRVCoherence(state)
  const journal     = useAlchemicalJournal(state)
  const sessionGoals = useSessionGoals(state)

  // Battery: Path A (Web BT) has numeric %; Path B has nothing.
  const battery = ble.battery

  // Signal quality: prefer SSE float, fall back to BLE electrode fraction
  const contactQuality: number | null =
    state?.contact_quality !== null && state?.contact_quality !== undefined
      ? state.contact_quality
      : ble.status === 'streaming'
        ? bleContactToQuality(ble.contact)
        : null

  // Stage 3 artifact gate fields (default-safe when state is null)
  const artifactRejected = state?.artifact_rejected ?? false
  const artifactReasons  = state?.artifact_reasons  ?? []
  const badChannels      = state?.bad_channels       ?? []
  const channelImpedances = state?.channel_impedances ?? {}

  // Shared derived data
  const eegSamples   = (state as any)?.eeg_samples ?? syntheticEEGSamples(state?.bands ?? null)
  const channelBands = syntheticChannelBands(state?.bands ?? null, 'alpha')

  return (
    <div style={S.app}>
      {/* ── Header ── */}
      <header style={S.header}>
        <div style={S.headerLeft}>
          <h1 style={S.title}>⚡ Neurolink</h1>
          <DeviceStatusBar
            battery={battery}
            contactQuality={contactQuality}
            poorContact={state?.poor_contact ?? false}
            artifactRejected={artifactRejected}
            artifactReasons={artifactReasons}
            source={connected ? (state?.source ?? null) : null}
            connected={connected}
          />
        </div>
        <span style={statusBadge(connected)}>
          <span style={dot(connected)} />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      {/* ── Tab bar ── */}
      <div style={S.tabBar}>
        {TABS.map(t => (
          <button key={t.id} style={tabBtn(t.id, tab)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════ LIVE TAB ════════════════════════════ */}
      {tab === 'live' && (
        <>
          <div style={{ ...S.grid, gridTemplateColumns: '1fr' }}>
            <div style={S.cardWide}>
              <div style={S.cardTitle}>Device Connection</div>
              <ConnectionPanel
                apiUrl={API_URL}
                connected={connected}
                bleInstance={ble}
              />
            </div>
          </div>

          <div style={{ ...S.grid, marginTop: 14 }}>
            <div style={S.card}>
              <div style={S.cardTitle}>Band Powers</div>
              <BandPowerChart bands={state?.bands ?? null} />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>Band Trends · 60 s</div>
              <BandTrend bands={state?.bands ?? null} baselineAlpha={null} />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>Focus &amp; Fatigue</div>
              <NeurofeedbackGauge
                focusScore={state?.focus_score ?? 0}
                fatigueScore={state?.fatigue_score ?? 0}
                focusState={state?.focus_state ?? 'unknown'}
              />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>S-Space / Alchemical Stage</div>
              <SSpaceDisplay
                region={state?.region ?? 'A'}
                stage={state?.alchemical_stage ?? 'Nigredo'}
                regionV01={state?.region_v01 ?? 'A'}
                stageV01={state?.alchemical_stage_v01 ?? 'Nigredo'}
              />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>EA-1 Eligibility</div>
              <EA1Score ea1={state?.ea1 ?? null} />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>Heart Rate &amp; HRV</div>
              <HRVPanel
                hrBpm={state?.hr_bpm ?? null}
                hrv={state?.hrv_rmssd ?? null}
                rrBpm={state?.rr_bpm ?? null}
              />
            </div>

            {/* Contact Quality — upgraded with bad_channels */}
            <div style={S.card}>
              <div style={S.cardTitle}>Contact Quality</div>
              <ContactQuality
                poorContact={state?.poor_contact ?? false}
                contactQuality={state?.contact_quality ?? null}
                badChannels={badChannels}
              />
            </div>

            {/* Electrode Impedance */}
            <div style={S.card}>
              <div style={S.cardTitle}>⚡ Electrode Impedance</div>
              <ImpedancePanel impedances={channelImpedances} />
            </div>

            {/* ── Mind-Wandering Detector ──────────────────────────────── */}
            <div style={S.card}>
              <div style={S.cardTitle}>🧠 Mind-Wandering Log</div>
              <WanderingLog detector={detector} />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>Breathing</div>
              <BreathingPanel rrBpm={state?.rr_bpm ?? null} />
            </div>

            {/* IMUPanel — upgraded with motion gate threshold */}
            <div style={S.card}>
              <div style={S.cardTitle}>Head Pose &amp; Motion</div>
              <IMUPanel
                pitchDeg={state?.pitch_deg ?? null}
                rollDeg={state?.roll_deg ?? null}
                motionRms={state?.motion_rms ?? null}
              />
            </div>

            {/* Stage 3 artifact stats */}
            <div style={S.card}>
              <div style={S.cardTitle}>🛡️ Artifact Stats · Stage 3</div>
              <ArtifactStatsPanel stats={artifactStats} connected={connected} />
            </div>

            <div style={S.card}>
              <div style={S.cardTitle}>Calibration</div>
              <CalibrationPanel apiUrl={API_URL} />
            </div>

            {/* ── Adaptive Audio Neurofeedback ─────────────────────────── */}
            <div style={S.card}>
              <div style={S.cardTitle}>🔔 Audio Neurofeedback</div>
              <AudioFeedbackPanel audio={audio} />
            </div>

            {/* ── Personal Baseline (live tab summary) ─────────────────── */}
            <div style={S.card}>
              <div style={S.cardTitle}>📈 Personal Baseline</div>
              <PersonalBaseline result={baseline} />
            </div>
          </div>
        </>
      )}

      {/* ═══════════════════════ PIPELINE TAB ════════════════════════════ */}
      {tab === 'pipeline' && (
        <div style={S.grid}>
          {/* Unified Stage 0-3 health panel */}
          <div style={S.cardWide}>
            <div style={S.cardTitle}>🔬 DSP Pipeline · Stage 0–3 Live Status</div>
            <SignalPipelinePanel state={state} stats={artifactStats} />
          </div>

          {/* Bad channel detail */}
          <div style={S.card}>
            <div style={S.cardTitle}>📡 Channel Health · Stage 0 / 2</div>
            <BadChannelPanel badChannels={badChannels} />
          </div>

          {/* Artifact stats detail */}
          <div style={S.card}>
            <div style={S.cardTitle}>🛡️ Artifact Rejection · Stage 3</div>
            <ArtifactStatsPanel stats={artifactStats} connected={connected} />
          </div>

          {/* IMU full view */}
          <div style={S.card}>
            <div style={S.cardTitle}>Head Pose &amp; Motion Gate</div>
            <IMUPanel
              pitchDeg={state?.pitch_deg ?? null}
              rollDeg={state?.roll_deg ?? null}
              motionRms={state?.motion_rms ?? null}
            />
          </div>

          {/* Contact quality + channel pills */}
          <div style={S.card}>
            <div style={S.cardTitle}>Electrode Contact</div>
            <ContactQuality
              poorContact={state?.poor_contact ?? false}
              contactQuality={state?.contact_quality ?? null}
              badChannels={badChannels}
            />
          </div>

          {/* Per-channel impedance */}
          <div style={S.card}>
            <div style={S.cardTitle}>⚡ Electrode Impedance · Per Channel</div>
            <ImpedancePanel impedances={channelImpedances} />
          </div>

          {/* ── Artifact Guide ── */}
          <div style={S.cardWide}>
            <div style={S.cardTitle}>🧠 Artifact Intelligence Guide</div>
            <ArtifactGuidePanel
              state={state}
              rejectRate={artifactStats.rejectRate}
              connected={connected}
            />
          </div>
        </div>
      )}

      {/* ═══════════════════════ SPECTROGRAM TAB ═════════════════════════ */}
      {tab === 'spectrogram' && (
        <div style={S.grid}>
          <div style={S.cardWide}>
            <div style={S.cardTitle}>Rolling Spectrogram · 30 s window · 1–50 Hz (log)</div>
            <RollingSpectrogram eegSamples={eegSamples} sampleRate={256} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Band Power Trends · 60 s</div>
            <BandTrend bands={state?.bands ?? null} baselineAlpha={null} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Current Band Powers</div>
            <BandPowerChart bands={state?.bands ?? null} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ TOPO MAP TAB ════════════════════════════ */}
      {tab === 'topo' && (
        <div style={S.grid}>
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={S.cardTitle}>Topographic Map · 4 Electrodes</div>
            <TopoMap bands={state?.bands ?? null} channelBands={channelBands} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Band Powers</div>
            <BandPowerChart bands={state?.bands ?? null} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Band Trends · 60 s</div>
            <BandTrend bands={state?.bands ?? null} baselineAlpha={null} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ CONNECTIVITY TAB ════════════════════════ */}
      {tab === 'connectivity' && (
        <div style={S.grid}>
          <div style={S.card}>
            <div style={S.cardTitle}>Channel Connectivity (PLV proxy)</div>
            <ConnectivityArc bands={state?.bands ?? null} channelBands={channelBands} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Focus &amp; Fatigue Gauges</div>
            <NeurofeedbackGauge
              focusScore={state?.focus_score ?? 0}
              fatigueScore={state?.fatigue_score ?? 0}
              focusState={state?.focus_state ?? 'unknown'}
            />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>Band Trends · 60 s</div>
            <BandTrend bands={state?.bands ?? null} baselineAlpha={null} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ PRACTICE TAB ════════════════════════════ */}
      {tab === 'practice' && (
        <div style={S.grid}>
          <div style={S.cardWide}>
            <div style={S.cardTitle}>🏆 Session Goals &amp; Achievements</div>
            <SessionGoals goals={sessionGoals} />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>🕐 Guided Meditation Timer</div>
            <MeditationTimer />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>💓 HRV Coherence Trainer</div>
            <HRVCoherenceTrainer
              result={coherence}
              hrBpm={state?.hr_bpm ?? null}
              hrv={state?.hrv_rmssd ?? null}
            />
          </div>
          <div style={S.card}>
            <div style={S.cardTitle}>📈 Personal Baseline Deviations</div>
            <PersonalBaseline result={baseline} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ JOURNAL TAB ═════════════════════════════ */}
      {tab === 'journal' && (
        <div style={S.grid}>
          <div style={S.cardWide}>
            <div style={S.cardTitle}>📓 Alchemical Stage Journal</div>
            <AlchemicalJournal journal={journal} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ HISTORY TAB ═════════════════════════════ */}
      {tab === 'history' && (
        <div style={S.grid}>
          <div style={S.cardWide}>
            <div style={S.cardTitle}>📊 Session History</div>
            <SessionHistoryPanel history={history} />
          </div>
        </div>
      )}

      {/* ═══════════════════════ FILTERS TAB ═════════════════════════════ */}
      {tab === 'filters' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* ── Section 1: Pipeline stage on/off toggles ── */}
          <FiltersPage apiUrl={API_URL} />

          {/* ── Divider ── */}
          <div style={S.divider}>
            <div style={S.dividerLine} />
            <span style={S.dividerLabel}>Stage 3 Thresholds</span>
            <div style={S.dividerLine} />
          </div>

          {/* ── Section 2: Stage 3 numeric threshold editor ── */}
          <ArtifactConfigPanel apiUrl={API_URL} />

          {/* ── Divider ── */}
          <div style={S.divider}>
            <div style={S.dividerLine} />
            <span style={S.dividerLabel}>Artifact Intelligence</span>
            <div style={S.dividerLine} />
          </div>

          {/* ── Section 3: Live artifact guide + encyclopedia ── */}
          <ArtifactGuidePanel
            state={state}
            rejectRate={artifactStats.rejectRate}
            connected={connected}
          />

        </div>
      )}

      {state && (
        <p style={S.frameCount}>Frame #{state.frame_count} · Source: {state.source}</p>
      )}
    </div>
  )
}
