/**
 * ConnectionPanel — full rewrite implementing Path A + Path B.
 *
 * Path A (Web Bluetooth tab):
 *   Uses useMuseBLE hook.  Works in Chrome/Edge/Opera only.
 *   No terminal, no drivers, no install required.
 *   The browser's native BLE picker handles device selection.
 *
 * Path B (Backend BLE tab):
 *   Calls POST /api/v1/neurolink/ble/scan to trigger a backend bleak
 *   scan, then lets the user pick a device from the scan results and
 *   calls POST /api/v1/neurolink/connect.  Valid when the backend
 *   process runs on the same machine as the Muse (local install).
 *   Also exposes the legacy mock + LSL adapter options.
 *
 * The active tab is remembered in component state only (no localStorage
 * — sandboxed iframe restriction).
 */
import React, { useState } from 'react'
import { useMuseBLE } from '../hooks/useMuseBLE'
import MuseConnectButton from './MuseConnectButton'
import type { ConnectRequest, ConnectResponse, BLEDevice } from '../types'

interface Props {
  apiUrl:    string
  connected: boolean
  onStatusChange?: (msg: string, ok: boolean) => void
}

type Tab = 'webbt' | 'backend'

// ── Styles ────────────────────────────────────────────────────────────────────
const s: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', gap: 16 },
  tabs: { display: 'flex', gap: 0, borderBottom: '1px solid #30363d' },
  tab: (active: boolean): React.CSSProperties => ({
    padding: '7px 18px',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    border: 'none',
    background: 'none',
    color: active ? '#58a6ff' : '#8b949e',
    borderBottom: active ? '2px solid #58a6ff' : '2px solid transparent',
    marginBottom: -1,
    transition: 'color 180ms ease',
  }),
  tabBody: { paddingTop: 4 },
  row: { display: 'flex', gap: 8, flexWrap: 'wrap' as const, alignItems: 'flex-end' },
  field: { display: 'flex', flexDirection: 'column', gap: 4, flex: 1, minWidth: 120 },
  label: { fontSize: 11, fontWeight: 600, color: '#8b949e', textTransform: 'uppercase' as const, letterSpacing: 0.8 },
  select: {
    background: '#0d1117', border: '1px solid #30363d', borderRadius: 6,
    color: '#e6edf3', padding: '6px 10px', fontSize: 13, cursor: 'pointer',
  },
  input: {
    background: '#0d1117', border: '1px solid #30363d', borderRadius: 6,
    color: '#e6edf3', padding: '6px 10px', fontSize: 13, flex: 2, minWidth: 160,
  },
  scanBtn: {
    padding: '7px 14px', borderRadius: 6, fontSize: 13, fontWeight: 600,
    cursor: 'pointer', border: '1px solid #30363d',
    background: 'rgba(139,148,158,0.15)', color: '#8b949e', whiteSpace: 'nowrap' as const,
  },
  btnConnect: {
    padding: '7px 18px', borderRadius: 6, fontSize: 13, fontWeight: 600,
    cursor: 'pointer', border: '1px solid #238636',
    background: 'rgba(46,160,67,0.15)', color: '#3fb950', whiteSpace: 'nowrap' as const,
  },
  btnDisconnect: {
    padding: '7px 18px', borderRadius: 6, fontSize: 13, fontWeight: 600,
    cursor: 'pointer', border: '1px solid #da3633',
    background: 'rgba(248,81,73,0.15)', color: '#f85149', whiteSpace: 'nowrap' as const,
  },
  btnDisabled: { opacity: 0.45, cursor: 'not-allowed' as const },
  statusOk:  { fontSize: 12, color: '#3fb950', marginTop: 2 },
  statusErr: { fontSize: 12, color: '#f85149', marginTop: 2 },
  deviceList: {
    display: 'flex', flexDirection: 'column' as const, gap: 6, marginTop: 4,
  },
  deviceRow: (selected: boolean): React.CSSProperties => ({
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '8px 12px', borderRadius: 6, cursor: 'pointer',
    border: `1px solid ${selected ? '#388bfd' : '#30363d'}`,
    background: selected ? 'rgba(56,139,253,0.1)' : 'rgba(22,27,34,0.6)',
    fontSize: 13, color: '#e6edf3', transition: 'all 180ms ease',
  }),
  deviceRssi: { marginLeft: 'auto', fontSize: 11, color: '#8b949e' },
  scanHint: { fontSize: 12, color: '#484f58', marginTop: 4 },
  pathBadge: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    padding: '2px 8px', borderRadius: 20, fontSize: 11, fontWeight: 600,
    background: 'rgba(56,139,253,0.12)', border: '1px solid rgba(56,139,253,0.3)',
    color: '#58a6ff',
  },
}

const ADAPTER_TYPES: ConnectRequest['adapter_type'][] = ['ble', 'lsl', 'mock']
const DEVICE_MODELS: ConnectRequest['device_model'][] = ['muse_s_gen1', 'muse_s_athena', 'mock']

// ── Backend BLE Tab ───────────────────────────────────────────────────────────
function BackendBLETab({
  apiUrl, connected, onStatusChange,
}: { apiUrl: string; connected: boolean; onStatusChange?: (m: string, ok: boolean) => void }) {
  const [adapterType, setAdapterType] = useState<ConnectRequest['adapter_type']>('ble')
  const [deviceModel, setDeviceModel] = useState<ConnectRequest['device_model']>('muse_s_gen1')
  const [manualAddress, setManualAddress]   = useState('')
  const [scanning,    setScanning]    = useState(false)
  const [scanResults, setScanResults] = useState<BLEDevice[]>([])
  const [selectedAddr, setSelectedAddr] = useState<string | null>(null)
  const [busy, setBusy]               = useState(false)
  const [statusMsg, setStatusMsg]     = useState<{ text: string; ok: boolean } | null>(null)

  const notify = (text: string, ok: boolean) => {
    setStatusMsg({ text, ok })
    onStatusChange?.(text, ok)
  }

  const handleScan = async () => {
    setScanning(true)
    setScanResults([])
    setSelectedAddr(null)
    setStatusMsg(null)
    try {
      const res = await fetch(`${apiUrl}/api/v1/neurolink/ble/scan`, { method: 'GET' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json() as { devices: BLEDevice[] }
      setScanResults(data.devices ?? [])
      if ((data.devices ?? []).length === 0) {
        notify('No Muse devices found. Make sure your headband is powered on.', false)
      } else {
        setStatusMsg({ text: `Found ${data.devices.length} device(s). Select one below.`, ok: true })
      }
    } catch (err) {
      notify(`Scan error: ${String(err)}`, false)
    } finally {
      setScanning(false)
    }
  }

  const handleConnect = async () => {
    setBusy(true)
    setStatusMsg(null)
    const address = selectedAddr || manualAddress.trim() || null
    try {
      const body: ConnectRequest = { adapter_type: adapterType, device_model: deviceModel, address }
      const res = await fetch(`${apiUrl}/api/v1/neurolink/connect`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Adapter / device pickers */}
      <div style={s.row}>
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
              else if (deviceModel === 'mock') setDeviceModel('muse_s_gen1')
            }}
          >
            {ADAPTER_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div style={s.field}>
          <label style={s.label}>Device</label>
          <select
            style={s.select}
            value={deviceModel}
            disabled={busy || connected}
            onChange={e => setDeviceModel(e.target.value as ConnectRequest['device_model'])}
          >
            {DEVICE_MODELS
              .filter(m => adapterType === 'mock' ? m === 'mock' : m !== 'mock')
              .map(m => <option key={m} value={m}>{m}</option>)
            }
          </select>
        </div>
      </div>

      {/* BLE-specific: scan button + manual address */}
      {adapterType === 'ble' && !connected && (
        <div style={s.row}>
          <button
            style={{ ...s.scanBtn, ...(scanning ? s.btnDisabled : {}) }}
            onClick={handleScan}
            disabled={scanning}
          >
            {scanning ? '⏳ Scanning…' : '🔍 Scan for Muse Devices'}
          </button>

          <div style={{ ...s.field, flex: 2 }}>
            <label style={s.label}>Manual BLE Address</label>
            <input
              type="text"
              style={s.input}
              placeholder="AA:BB:CC:DD:EE:FF (optional)"
              value={manualAddress}
              disabled={busy}
              onChange={e => { setManualAddress(e.target.value); setSelectedAddr(null) }}
            />
          </div>
        </div>
      )}

      {/* Scan results */}
      {scanResults.length > 0 && (
        <div style={s.deviceList}>
          {scanResults.map(dev => (
            <div
              key={dev.address}
              style={s.deviceRow(selectedAddr === dev.address)}
              onClick={() => { setSelectedAddr(dev.address); setManualAddress('') }}
              role="button"
              tabIndex={0}
              onKeyDown={e => e.key === 'Enter' && setSelectedAddr(dev.address)}
            >
              <span>🧠</span>
              <span style={{ fontWeight: 600 }}>{dev.name ?? 'Muse'}</span>
              <span style={{ color: '#8b949e', fontSize: 12 }}>{dev.address}</span>
              {dev.rssi !== undefined && (
                <span style={s.deviceRssi}>RSSI {dev.rssi} dBm</span>
              )}
            </div>
          ))}
          <p style={s.scanHint}>Click a device to select it, then press Connect below.</p>
        </div>
      )}

      {/* Connect / Disconnect */}
      <div style={s.row}>
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
        <p style={statusMsg.ok ? s.statusOk : s.statusErr}>{statusMsg.text}</p>
      )}
    </div>
  )
}

// ── Main ConnectionPanel ──────────────────────────────────────────────────────
export default function ConnectionPanel({ apiUrl, connected, onStatusChange }: Props) {
  const [tab, setTab] = useState<Tab>('webbt')
  const ble = useMuseBLE(apiUrl)

  return (
    <div style={s.root}>
      {/* Tab bar */}
      <div style={s.tabs}>
        <button style={s.tab(tab === 'webbt')} onClick={() => setTab('webbt')}>
          🧠 Web Bluetooth
          <span style={{ marginLeft: 6, ...s.pathBadge }}>Path A</span>
        </button>
        <button style={s.tab(tab === 'backend')} onClick={() => setTab('backend')}>
          🖥 Backend BLE
          <span style={{ marginLeft: 6, ...s.pathBadge }}>Path B</span>
        </button>
      </div>

      <div style={s.tabBody}>
        {tab === 'webbt' && (
          <MuseConnectButton
            status={ble.status}
            deviceName={ble.deviceName}
            battery={ble.battery}
            contact={ble.contact}
            errorMsg={ble.errorMsg}
            framesSent={ble.framesSent}
            onConnect={ble.connect}
            onDisconnect={ble.disconnect}
          />
        )}

        {tab === 'backend' && (
          <BackendBLETab
            apiUrl={apiUrl}
            connected={connected}
            onStatusChange={onStatusChange}
          />
        )}
      </div>
    </div>
  )
}
