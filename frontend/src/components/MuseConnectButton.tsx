/**
 * MuseConnectButton — single-component, foolproof Path A (Web Bluetooth) UI.
 *
 * Rendered inside ConnectionPanel when Path A tab is active.
 * Shows exactly one primary action at a time. The browser compat guard
 * appears immediately on unsupported browsers so the user never wastes
 * a click.
 */
import React from 'react'
import type { ContactQuality, BLEStatus } from '../hooks/useMuseBLE'

interface Props {
  status:     BLEStatus
  deviceName: string | null
  battery:    number | null
  contact:    ContactQuality
  errorMsg:   string | null
  framesSent: number
  onConnect:    () => void
  onDisconnect: () => void
}

const ELECTRODE_LABELS: { key: keyof ContactQuality; label: string }[] = [
  { key: 'tp9',  label: 'TP9'  },
  { key: 'af7',  label: 'AF7'  },
  { key: 'af8',  label: 'AF8'  },
  { key: 'tp10', label: 'TP10' },
]

const s: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  unsupportedBanner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
    padding: '14px 16px',
    background: 'rgba(210,153,34,0.12)',
    border: '1px solid rgba(210,153,34,0.35)',
    borderRadius: 8,
  },
  unsupportedIcon: {
    fontSize: 22,
    lineHeight: 1,
    flexShrink: 0,
    marginTop: 1,
  },
  unsupportedText: {
    fontSize: 13,
    color: '#e3b341',
    lineHeight: 1.55,
  },
  unsupportedBold: {
    fontWeight: 700,
    display: 'block',
    marginBottom: 4,
  },
  chromeLink: {
    color: '#58a6ff',
    textDecoration: 'none',
    fontWeight: 600,
  },
  bigBtn: (active: boolean, danger: boolean): React.CSSProperties => ({
    width: '100%',
    padding: '13px 0',
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 700,
    cursor: active ? 'pointer' : 'not-allowed',
    opacity: active ? 1 : 0.5,
    border: `1px solid ${danger ? '#da3633' : '#238636'}`,
    background: danger ? 'rgba(248,81,73,0.15)' : 'rgba(46,160,67,0.15)',
    color: danger ? '#f85149' : '#3fb950',
    transition: 'opacity 180ms ease',
    letterSpacing: '0.3px',
  }),
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    fontSize: 13,
    color: '#8b949e',
  },
  statusDot: (streaming: boolean): React.CSSProperties => ({
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
    background: streaming ? '#3fb950' : '#484f58',
    boxShadow: streaming ? '0 0 6px #3fb950' : 'none',
  }),
  deviceName: {
    fontWeight: 600,
    color: '#e6edf3',
  },
  battery: {
    marginLeft: 'auto',
    fontSize: 12,
    color: '#8b949e',
  },
  electrodesRow: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap' as const,
  },
  electrodePill: (good: boolean, active: boolean): React.CSSProperties => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '3px 10px',
    borderRadius: 20,
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 0.5,
    border: `1px solid ${!active ? '#30363d' : good ? '#238636' : '#da3633'}`,
    background: !active ? 'transparent' : good ? 'rgba(46,160,67,0.1)' : 'rgba(248,81,73,0.1)',
    color: !active ? '#484f58' : good ? '#3fb950' : '#f85149',
    transition: 'all 300ms ease',
  }),
  electrodeDot: (good: boolean, active: boolean): React.CSSProperties => ({
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: !active ? '#484f58' : good ? '#3fb950' : '#f85149',
  }),
  hintText: {
    fontSize: 12,
    color: '#484f58',
    lineHeight: 1.6,
  },
  errorText: {
    fontSize: 12,
    color: '#f85149',
  },
  framesText: {
    fontSize: 11,
    color: '#484f58',
    textAlign: 'right' as const,
  },
}

const STATUS_MESSAGES: Record<BLEStatus, string> = {
  unsupported: 'Web Bluetooth not supported',
  idle:        'Ready to connect',
  requesting:  'Browser device picker is open — select your Muse…',
  connecting:  'Connecting to headband…',
  streaming:   'Streaming',
  reconnecting:'Reconnecting…',
  error:       'Connection error',
}

export default function MuseConnectButton({
  status, deviceName, battery, contact, errorMsg, framesSent,
  onConnect, onDisconnect,
}: Props) {
  const isStreaming    = status === 'streaming'
  const isBusy        = status === 'requesting' || status === 'connecting' || status === 'reconnecting'
  const isUnsupported = status === 'unsupported'
  const isIdle        = status === 'idle' || status === 'error'
  const electrodesActive = isStreaming || status === 'reconnecting'

  if (isUnsupported) {
    return (
      <div style={s.root}>
        <div style={s.unsupportedBanner}>
          <span style={s.unsupportedIcon}>⚠️</span>
          <div style={s.unsupportedText}>
            <strong style={s.unsupportedBold}>Web Bluetooth is not supported in this browser.</strong>
            To connect your Muse headband directly, please open this page in{' '}
            <a
              href="https://www.google.com/chrome/"
              target="_blank"
              rel="noopener noreferrer"
              style={s.chromeLink}
            >Google Chrome</a>
            {' '}or{' '}
            <a
              href="https://www.microsoft.com/en-us/edge"
              target="_blank"
              rel="noopener noreferrer"
              style={s.chromeLink}
            >Microsoft Edge</a>
            . Firefox and Safari do not support Web Bluetooth.
            <br /><br />
            Alternatively, use the <strong>Backend BLE</strong> tab if you are
            running Neurolink locally on the same machine as your headband.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={s.root}>
      {/* Primary action button */}
      {isStreaming ? (
        <button
          style={s.bigBtn(true, true)}
          onClick={onDisconnect}
        >
          ⏹ Disconnect Headband
        </button>
      ) : (
        <button
          style={s.bigBtn(!isBusy, false)}
          onClick={!isBusy ? onConnect : undefined}
          disabled={isBusy}
        >
          {isBusy ? '⏳ ' : '🧠 '}
          {isBusy
            ? (status === 'requesting' ? 'Waiting for device picker…' :
               status === 'reconnecting' ? 'Reconnecting…' :
               'Connecting…')
            : 'Connect Muse Headband'
          }
        </button>
      )}

      {/* Status row */}
      <div style={s.statusRow}>
        <span style={s.statusDot(isStreaming)} />
        <span>{STATUS_MESSAGES[status]}</span>
        {deviceName && <span style={s.deviceName}>{deviceName}</span>}
        {battery !== null && (
          <span style={s.battery}>🔋 {battery}%</span>
        )}
      </div>

      {/* Electrode contact quality — only shown once we have a device */}
      <div style={s.electrodesRow}>
        {ELECTRODE_LABELS.map(({ key, label }) => (
          <span key={key} style={s.electrodePill(contact[key], electrodesActive)}>
            <span style={s.electrodeDot(contact[key], electrodesActive)} />
            {label}
          </span>
        ))}
      </div>

      {/* Context hints */}
      {isIdle && !errorMsg && (
        <p style={s.hintText}>
          Put your Muse S headband on and press its power button until the light
          pulses. Then click <em>Connect Muse Headband</em> above and select
          your device from the browser popup.
        </p>
      )}

      {status === 'requesting' && (
        <p style={s.hintText}>
          A device picker should have appeared. If you don't see it, check if
          your browser blocked the popup, then try again.
        </p>
      )}

      {errorMsg && (
        <p style={s.errorText}>⚠ {errorMsg}</p>
      )}

      {framesSent > 0 && (
        <p style={s.framesText}>{framesSent} frame{framesSent !== 1 ? 's' : ''} sent to backend</p>
      )}
    </div>
  )
}
