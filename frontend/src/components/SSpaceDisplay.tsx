import React from 'react'

interface Props {
  region: string
  stage: string
  regionV01: string
  stageV01: string
}

const STAGE_COLORS: Record<string, string> = {
  Nigredo: '#6e7681',
  Albedo: '#388bfd',
  Citrinitas: '#d29922',
  Rubedo: '#f85149',
  Multiplicatio: '#bc8cff',
  Coagulatio: '#8b949e',
  Sublimatio: '#3fb950',
  Solutio: '#58a6ff',
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 12 },
  row: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  label: { fontSize: 12, color: '#8b949e' },
  badge: (color: string): React.CSSProperties => ({
    display: 'inline-block',
    padding: '3px 10px',
    borderRadius: 12,
    background: `${color}22`,
    color,
    border: `1px solid ${color}66`,
    fontSize: 13,
    fontWeight: 600,
  }),
  region: { fontSize: 28, fontWeight: 700, color: '#e6edf3' },
  divider: { height: 1, background: '#21262d', margin: '4px 0' },
}

export default function SSpaceDisplay({ region, stage, regionV01, stageV01 }: Props) {
  const color = STAGE_COLORS[stage] ?? '#58a6ff'
  const colorV01 = STAGE_COLORS[stageV01] ?? '#58a6ff'

  return (
    <div style={styles.container}>
      <div style={styles.row}>
        <div>
          <div style={styles.label}>v2 Alchemical</div>
          <span style={{ ...styles.region, color }}>{region}</span>
        </div>
        <span style={styles.badge(color)}>{stage}</span>
      </div>

      <div style={styles.divider} />

      <div style={styles.row}>
        <div>
          <div style={styles.label}>v0.1 S-Space</div>
          <span style={{ fontSize: 20, fontWeight: 600, color: colorV01 }}>{regionV01}</span>
        </div>
        <span style={styles.badge(colorV01)}>{stageV01}</span>
      </div>
    </div>
  )
}
