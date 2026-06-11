import React from 'react'
import { useNeurolinkSSE } from './hooks/useNeurolinkSSE'
import BandPowerChart from './components/BandPowerChart'
import SSpaceDisplay from './components/SSpaceDisplay'
import EA1Score from './components/EA1Score'
import HRVPanel from './components/HRVPanel'
import FocusFatigueGauge from './components/FocusFatigueGauge'
import ContactQuality from './components/ContactQuality'

const API_URL = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000'

const styles: Record<string, React.CSSProperties> = {
  app: {
    maxWidth: 1200,
    margin: '0 auto',
    padding: '24px 16px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
    paddingBottom: 16,
    borderBottom: '1px solid #30363d',
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    color: '#58a6ff',
    letterSpacing: '-0.5px',
  },
  statusBadge: (
    connected: boolean
  ): React.CSSProperties => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 12px',
    borderRadius: 20,
    fontSize: 13,
    fontWeight: 600,
    background: connected ? 'rgba(46,160,67,0.15)' : 'rgba(248,81,73,0.15)',
    color: connected ? '#3fb950' : '#f85149',
    border: `1px solid ${connected ? '#238636' : '#da3633'}`,
  }),
  dot: (connected: boolean): React.CSSProperties => ({
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: connected ? '#3fb950' : '#f85149',
  }),
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: 16,
  },
  card: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 12,
    padding: 20,
  },
  cardTitle: {
    fontSize: 13,
    fontWeight: 600,
    color: '#8b949e',
    textTransform: 'uppercase' as const,
    letterSpacing: 1,
    marginBottom: 14,
  },
  frameCount: {
    fontSize: 12,
    color: '#484f58',
    marginTop: 16,
    textAlign: 'center' as const,
  },
}

export default function App() {
  const state = useNeurolinkSSE(`${API_URL}/api/v1/neurolink/stream`)
  const connected = state?.connected ?? false

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>⚡ Neurolink</h1>
        <span style={styles.statusBadge(connected)}>
          <span style={styles.dot(connected)} />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Band Powers</div>
          <BandPowerChart bands={state?.bands ?? null} />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>S-Space / Alchemical Stage</div>
          <SSpaceDisplay
            region={state?.region ?? 'A'}
            stage={state?.alchemical_stage ?? 'Nigredo'}
            regionV01={state?.region_v01 ?? 'A'}
            stageV01={state?.alchemical_stage_v01 ?? 'Nigredo'}
          />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>EA-1 Eligibility</div>
          <EA1Score ea1={state?.ea1 ?? null} />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>Heart Rate & HRV</div>
          <HRVPanel
            hrBpm={state?.hr_bpm ?? null}
            hrv={state?.hrv_rmssd ?? null}
            rrBpm={state?.rr_bpm ?? null}
          />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>Focus & Fatigue</div>
          <FocusFatigueGauge
            focusState={state?.focus_state ?? 'unknown'}
            focusScore={state?.focus_score ?? 0}
            fatigueScore={state?.fatigue_score ?? 0}
          />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>Contact Quality</div>
          <ContactQuality
            poorContact={state?.poor_contact ?? false}
            contactQuality={state?.contact_quality ?? null}
          />
        </div>
      </div>

      {state && (
        <p style={styles.frameCount}>Frame #{state.frame_count} · Source: {state.source}</p>
      )}
    </div>
  )
}
