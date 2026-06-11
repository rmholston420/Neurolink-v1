/**
 * ConnectionPanel — lets the user pick adapter type, device model, and
 * optional BLE address, then POST /api/v1/neurolink/connect or /disconnect.
 */
import React, { useState } from 'react'
import type { ConnectRequest, ConnectResponse } from '../types'

interface Props {
  apiUrl: string
  connected: boolean
  onStatusChange?: (msg: string, ok: boolean) => void
}

const ADAPTER_TYPES: ConnectRequest['adapter_type'][] = ['mock', 'ble', 'lsl']
const DEVICE_MODELS: ConnectRequest['device_model'][] = [
  'mock',
  'muse_s_gen1',
  'muse_s_athena',
]

const s: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', gap: 12 },
  row: { display: 'flex', gap: 8, flexWrap: 'wrap' as const, alignItems: 'flex-end' },
  field: { display: 'flex', flexDirection: 'column', gap: 4, flex: 1, minWidth: 120 },
  label: { fontSize: 11, fontWeight: 600, color: '#8b949e', textTransform: 'uppercase' as const, letterSpacing: 0.8 },
  select: {
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 6,
    color: '#e6edf3',
    padding: '6px 10px',
    fontSize: 13,
    cursor: 'pointer',
  },
  input: {
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 6,
    color: '#e6edf3',
    padding: '6px 10px',
    fontSize: 13,
    flex: 2,
    minWidth: 160,
  },
  btnConnect: {
    padding: '7px 18px',
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    border: '1px solid #238636',
    background: 'rgba(46,160,67,0.15)',
    color: '#3fb950',
    whiteSpace: 'nowrap' as const,
  },
  btnDisconnect: {
    padding: '7px 18px',
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    border: '1px solid #da3633',
    background: 'rgba(248,81,73,0.15)',
    color: '#f85149',
    whiteSpace: 'nowrap' as const,
  },
  btnDisabled: {
    opacity: 0.45,
    cursor: 'not-allowed' as const,
  },
  status: (ok: boolean): React.CSSProperties => ({
    fontSize: 12,
    color: ok ? '#3fb950' : '#f85149',
    marginTop: 2,
  }),
}

export default function ConnectionPanel({ apiUrl, connected, onStatusChange }: Props) {
  const [adapterType, setAdapterType] = useState<ConnectRequest['adapter_type']>('mock')
  const [deviceModel, setDeviceModel] = useState<ConnectRequest['device_model']>('mock')
  const [address, setAddress] = useState('')
  const [busy, setBusy] = useState(false)
  const [statusMsg, setStatusMsg] = useState<{ text: string; ok: boolean } | null>(null)

  const notify = (text: string, ok: boolean) => {
    setStatusMsg({ text, ok })
    onStatusChange?.(text, ok)
  }

  const handleConnect = async () => {
    setBusy(true)
    setStatusMsg(null)
    try {
      const body: ConnectRequest = {
        adapter_type: adapterType,
        device_model: deviceModel,
        address: address.trim() || null,
      }
      const res = await fetch(`${apiUrl}/api/v1/neurolink/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data: ConnectResponse = await res.json()
      notify(data.message ?? (data.ok ? 'Connected' : 'Failed'), data.ok)
    } catch (err) {
      notify(`Network error: ${String(err)}`, false)
    } finally {
      setBusy(false)
    }
  }

  const handleDisconnect = async () => {
    setBusy(true)
    setStatusMsg(null)
    try {
      const res = await fetch(`${apiUrl}/api/v1/neurolink/disconnect`, { method: 'POST' })
      const data = await res.json()
      notify(data.message ?? 'Disconnected', true)
    } catch (err) {
      notify(`Network error: ${String(err)}`, false)
    } finally {
      setBusy(false)
    }
  }

  const needsAddress = adapterType === 'ble'

  return (
    <div style={s.root}>
      <div style={s.row}>
        {/* Adapter type */}
        <div style={s.field}>
          <label style={s.label}>Adapter</label>
          <select
            style={s.select}
            value={adapterType}
            disabled={busy || connected}
            onChange={e => {
              const v = e.target.value as ConnectRequest['adapter_type']
              setAdapterType(v)
              if (v === 'mock') setDeviceModel('mock')
            }}
          >
            {ADAPTER_TYPES.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Device model */}
        <div style={s.field}>
          <label style={s.label}>Device</label>
          <select
            style={s.select}
            value={deviceModel}
            disabled={busy || connected}
            onChange={e => setDeviceModel(e.target.value as ConnectRequest['device_model'])}
          >
            {DEVICE_MODELS.filter(m => adapterType === 'mock' ? m === 'mock' : m !== 'mock').map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* BLE address — only shown when adapter = ble */}
        {needsAddress && (
          <div style={{ ...s.field, flex: 2 }}>
            <label style={s.label}>BLE Address</label>
            <input
              type="text"
              style={s.input}
              placeholder="AA:BB:CC:DD:EE:FF"
              value={address}
              disabled={busy || connected}
              onChange={e => setAddress(e.target.value)}
            />
          </div>
        )}

        {/* Connect / Disconnect button */}
        {!connected ? (
          <button
            style={{ ...s.btnConnect, ...(busy ? s.btnDisabled : {}) }}
            onClick={handleConnect}
            disabled={busy}
          >
            {busy ? 'Connecting…' : 'Connect'}
          </button>
        ) : (
          <button
            style={{ ...s.btnDisconnect, ...(busy ? s.btnDisabled : {}) }}
            onClick={handleDisconnect}
            disabled={busy}
          >
            {busy ? 'Disconnecting…' : 'Disconnect'}
          </button>
        )}
      </div>

      {statusMsg && (
        <p style={s.status(statusMsg.ok)}>{statusMsg.text}</p>
      )}
    </div>
  )
}
