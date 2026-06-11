import React from 'react'

interface Props {
  /** v2 classifier: single-letter region label (e.g. 'A'–'F') */
  region: string
  /** v2 alchemical stage name */
  stage: string
  /** v0.1 S-Space region label */
  regionV01: string
  /** v0.1 alchemical stage name */
  stageV01: string
}

// Map stage names to accent colours – unknown stages fall back to neutral.
const STAGE_COLOURS: Record<string, string> = {
  Nigredo: '#6e7681',
  Albedo: '#79c0ff',
  Citrinitas: '#e3b341',
  Rubedo: '#ff7b72',
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
    color: '#484f58',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  regionBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 36,
    height: 36,
    borderRadius: 6,
    background: '#21262d',
    border: '1px solid #30363d',
    fontSize: 18,
    fontWeight: 700,
    color: '#e6edf3',
    flexShrink: 0,
  },
  divider: {
    height: 1,
    background: '#21262d',
  },
}

// Standalone helper — NOT inside the styles Record so it can be called as a function.
function stageBadgeStyle(colour: string): React.CSSProperties {
  return {
    display: 'inline-block',
    padding: '2px 10px',
    borderRadius: 20,
    fontSize: 13,
    fontWeight: 600,
    color: colour,
    background: colour + '22',
    border: `1px solid ${colour}55`,
  }
}

function Section({
  label,
  region,
  stage,
}: {
  label: string
  region: string
  stage: string
}) {
  const colour = STAGE_COLOURS[stage] ?? '#8b949e'
  return (
    <div style={styles.section}>
      <span style={styles.sectionLabel}>{label}</span>
      <div style={styles.row}>
        <span style={styles.regionBadge}>{region}</span>
        <span style={stageBadgeStyle(colour)}>{stage}</span>
      </div>
    </div>
  )
}

export default function SSpaceDisplay({ region, stage, regionV01, stageV01 }: Props) {
  return (
    <div style={styles.container}>
      <Section label="v2 Alchemical" region={region} stage={stage} />
      <div style={styles.divider} />
      <Section label="v0.1 S-Space" region={regionV01} stage={stageV01} />
    </div>
  )
}
