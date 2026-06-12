/**
 * ArtifactBadge
 *
 * Displays a per-frame EEG artifact quality indicator sourced from
 * the Stage 3 ArtifactGate decision in the SSE stream.
 *
 * States
 * ------
 *   disconnected  — renders nothing
 *   clean frame   — small solid green dot (non-intrusive)
 *   rejected frame — amber shield icon with animated pulse + tooltip
 *                    listing the rejection cause(s)
 *
 * The badge is intentionally minimal so it doesn't compete visually
 * with the signal-quality and battery indicators already in DeviceStatusBar.
 */
import React from 'react'

interface Props {
  connected:        boolean
  artifactRejected: boolean
  artifactReasons:  string[]
}

// Human-readable labels for reason strings coming from artifact_gate.py
const REASON_LABELS: Record<string, string> = {
  amplitude: 'Amplitude spike',
  motion:    'Head movement',
  kurtosis:  'Muscle burst',
}

function formatReasons(reasons: string[]): string {
  if (reasons.length === 0) return 'Frame rejected'
  return reasons
    .map(r => REASON_LABELS[r] ?? r)
    .join(' \u00b7 ')  // middle dot separator
}

export default function ArtifactBadge({ connected, artifactRejected, artifactReasons }: Props) {
  if (!connected) return null

  if (!artifactRejected) {
    // Clean frame — unobtrusive green dot
    return (
      <div
        title="Signal clean"
        aria-label="EEG signal clean"
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: '#3fb950',
          flexShrink: 0,
          transition: 'background 400ms ease',
        }}
      />
    )
  }

  // Rejected frame — amber shield with pulse animation
  const tooltip = formatReasons(artifactReasons)

  return (
    <>
      <style>{`
        @keyframes nlArtifact {
          0%, 100% { opacity: 1;    transform: scale(1); }
          50%       { opacity: 0.5; transform: scale(0.88); }
        }
      `}</style>
      <div
        title={tooltip}
        aria-label={`EEG artifact detected: ${tooltip}`}
        role="status"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          animation: 'nlArtifact 1.1s ease-in-out infinite',
          flexShrink: 0,
        }}
      >
        {/* Shield SVG — inline so no external dependency */}
        <svg
          width="13"
          height="13"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
          style={{ flexShrink: 0 }}
        >
          <path
            d="M8 1L2 3.5V8c0 3.3 2.5 5.7 6 7 3.5-1.3 6-3.7 6-7V3.5L8 1z"
            fill="rgba(227,179,65,0.18)"
            stroke="#e3b341"
            strokeWidth="1.2"
            strokeLinejoin="round"
          />
          <line
            x1="8" y1="5.5"
            x2="8" y2="9"
            stroke="#e3b341"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
          <circle cx="8" cy="11" r="0.8" fill="#e3b341" />
        </svg>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: '#e3b341',
            letterSpacing: 0.4,
            textTransform: 'uppercase',
          }}
        >
          Artifact
        </span>
      </div>
    </>
  )
}
