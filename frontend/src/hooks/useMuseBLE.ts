/**
 * useMuseBLE — Path A: Web Bluetooth connection to Muse S Gen 1.
 *
 * Uses the Web Bluetooth API (Chrome/Edge/Opera only) to:
 *   1. Let the user pick their Muse from the browser's native BLE picker.
 *   2. Open a GATT connection and subscribe to EEG + control characteristics.
 *   3. Decode incoming 12-byte EEG packets into μV band samples.
 *   4. Batch-assemble 256-Hz frames and POST IngestPayload JSON to the
 *      backend /api/v1/neurolink/ingest endpoint.
 *   5. Expose electrode contact quality (AF7, AF8, TP9, TP10).
 *   6. Auto-reconnect on GATT disconnect with exponential backoff.
 *
 * Muse S Gen 1 GATT surface:
 *   Service:    0xfe8d  (InterAxon)
 *   Control:    273e0001-4c4d-454d-96be-f03bac821358  (write commands)
 *   EEG:        273e0003-4c4d-454d-96be-f03bac821358  (TP9 left ear)
 *   EEG:        273e0004-4c4d-454d-96be-f03bac821358  (AF7 front-left)
 *   EEG:        273e0005-4c4d-454d-96be-f03bac821358  (AF8 front-right)
 *   EEG:        273e0006-4c4d-454d-96be-f03bac821358  (TP10 right ear)
 *   Telemetry:  273e000b-4c4d-454d-96be-f03bac821358  (battery, contact)
 *
 * Packet format (12 bytes):
 *   [0..1]  uint16be  sequence number
 *   [2..3]  12-bit sample 0  (shifted right by the packing bit)
 *   [4..5]  12-bit sample 1
 *   ...                      (5 samples total per channel per packet)
 *   Scale: (raw_int - 2048) * 0.48828125  →  microvolts
 *
 * CMD_DATA is the 3-byte command that starts streaming:
 *   0x02, 0x64, 0x0a
 *
 * Secure-context note:
 *   Chrome/Edge only expose navigator.bluetooth on secure origins:
 *     - https://
 *     - http://localhost  (the literal hostname, NOT http://127.0.0.1)
 *   Opening the app at http://127.0.0.1:xxxx will make 'bluetooth' absent
 *   from navigator even on Chrome. window.isSecureContext is the canonical
 *   browser flag for this. We surface this as the 'insecure_origin' status
 *   so the UI can show a specific, actionable error rather than the generic
 *   "browser not supported" warning.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

// ── GATT UUIDs ────────────────────────────────────────────────────────────────
const MUSE_SERVICE        = 0xfe8d
const CHAR_CONTROL        = '273e0001-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_TP9        = '273e0003-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_AF7        = '273e0004-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_AF8        = '273e0005-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_TP10       = '273e0006-4c4d-454d-96be-f03bac821358'
const CHAR_TELEMETRY      = '273e000b-4c4d-454d-96be-f03bac821358'

// CMD_DATA: tell the Muse to start streaming EEG + telemetry
const CMD_DATA = new Uint8Array([0x02, 0x64, 0x0a])

// ── Packet decode ─────────────────────────────────────────────────────────────
// Each 12-byte EEG notification carries 5 x 12-bit samples.
// Bit-layout: 2 bytes seq, then 10 bytes of packed 12-bit ints (5 samples).
function decodeSamples(buf: DataView): number[] {
  const out: number[] = []
  // Samples start at byte 2, each occupies 1.5 bytes (12 bits)
  for (let i = 0; i < 5; i++) {
    const byteOffset = 2 + Math.floor(i * 12 / 8)
    const bitShift   = (i % 2 === 0) ? 4 : 0
    const hi = buf.getUint8(byteOffset)
    const lo = buf.getUint8(byteOffset + 1)
    const raw = ((hi << 8 | lo) >> bitShift) & 0xfff
    out.push((raw - 2048) * 0.48828125)   // → µV
  }
  return out
}

// ── Telemetry decode ──────────────────────────────────────────────────────────
// Byte 0: seq, Byte 1: battery (0-100), Byte 5..8: contact bits
function decodeTelemetry(buf: DataView): { battery: number; contact: boolean[] } {
  const battery = buf.byteLength > 1 ? Math.round((buf.getUint8(1) >> 1) * 100 / 63) : 0
  // Contact quality: 4 bits in byte 5 for TP9, AF7, AF8, TP10
  const contactByte = buf.byteLength > 5 ? buf.getUint8(5) : 0
  const contact = [
    (contactByte & 0x10) === 0,  // TP9
    (contactByte & 0x20) === 0,  // AF7
    (contactByte & 0x40) === 0,  // AF8
    (contactByte & 0x80) === 0,  // TP10
  ]
  return { battery, contact }
}

// ── Types ─────────────────────────────────────────────────────────────────────
export type BLEStatus =
  | 'unsupported'     // browser has no Web Bluetooth API at all (Firefox, Safari)
  | 'insecure_origin' // Chrome on http://127.0.0.1 — needs http://localhost instead
  | 'idle'            // not connected, ready to try
  | 'requesting'      // browser picker open
  | 'connecting'      // GATT negotiation in progress
  | 'streaming'       // data flowing
  | 'reconnecting'    // dropped, retrying
  | 'error'           // fatal or user-cancelled

export interface ContactQuality {
  tp9:  boolean
  af7:  boolean
  af8:  boolean
  tp10: boolean
}

export interface UseMuseBLEReturn {
  status:     BLEStatus
  deviceName: string | null
  battery:    number | null
  contact:    ContactQuality
  errorMsg:   string | null
  framesSent: number
  connect:    () => Promise<void>
  disconnect: () => Promise<void>
}

// ── Secure-context detection ──────────────────────────────────────────────────
// window.isSecureContext is the canonical browser flag:
//   true  on https://, http://localhost, file://
//   false on http://127.0.0.1, http://192.168.x.x, etc.
// We check this BEFORE looking for navigator.bluetooth so we can give a
// specific actionable message rather than a generic "not supported" error.
function detectBLEStatus(): BLEStatus {
  if (typeof window === 'undefined') return 'unsupported'
  if (!window.isSecureContext) return 'insecure_origin'
  if (typeof navigator === 'undefined' || !('bluetooth' in navigator)) return 'unsupported'
  return 'idle'
}

// ── Frame accumulator ─────────────────────────────────────────────────────────
// Accumulate raw µV samples from all 4 channels, then compute band-power
// estimates via a simple windowed RMS and POST to the backend.
const FRAME_SAMPLES = 256   // 1-second window at 256 Hz

interface ChannelBuffer {
  tp9:  number[]
  af7:  number[]
  af8:  number[]
  tp10: number[]
}

function rms(arr: number[]): number {
  if (arr.length === 0) return 0
  const sum = arr.reduce((s, v) => s + v * v, 0)
  return Math.sqrt(sum / arr.length)
}

// Trivial bandpower proxies from raw µV via RMS over the last window.
// Real bandpower requires FFT; the backend's DSP layer will recompute
// from the raw samples. We send the raw channel averages so the backend
// has real data even without its own FFT on the ingest path.
function estimateBandPowers(samples: number[]) {
  const r = rms(samples)
  // These weights mimic the mock adapter distribution and will be
  // overwritten by the backend's Welch estimator once it has raw samples.
  return {
    delta: r * 0.35,
    theta: r * 0.25,
    alpha: r * 0.20,
    beta:  r * 0.15,
    gamma: r * 0.05,
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useMuseBLE(apiUrl: string): UseMuseBLEReturn {
  const initialStatus = detectBLEStatus()
  const isUsable = initialStatus === 'idle'

  const [status,     setStatus]     = useState<BLEStatus>(initialStatus)
  const [deviceName, setDeviceName] = useState<string | null>(null)
  const [battery,    setBattery]    = useState<number | null>(null)
  const [contact,    setContact]    = useState<ContactQuality>({ tp9: false, af7: false, af8: false, tp10: false })
  const [errorMsg,   setErrorMsg]   = useState<string | null>(null)
  const [framesSent, setFramesSent] = useState(0)

  const deviceRef   = useRef<BluetoothDevice | null>(null)
  const serverRef   = useRef<BluetoothRemoteGATTServer | null>(null)
  const aliveRef    = useRef(true)
  const reconnDelay = useRef(2_000)
  const bufRef      = useRef<ChannelBuffer>({ tp9: [], af7: [], af8: [], tp10: [] })
  const frameCountRef = useRef(0)

  // POST assembled frame to backend
  const postFrame = useCallback(async (buf: ChannelBuffer) => {
    const allSamples = [
      ...buf.tp9.slice(-FRAME_SAMPLES),
      ...buf.af7.slice(-FRAME_SAMPLES),
      ...buf.af8.slice(-FRAME_SAMPLES),
      ...buf.tp10.slice(-FRAME_SAMPLES),
    ]
    const bp = estimateBandPowers(allSamples)
    const payload = {
      source: 'ble_webbt',
      ts: Date.now() / 1000,
      bands: bp,
      raw_eeg: {
        tp9:  buf.tp9.slice(-FRAME_SAMPLES),
        af7:  buf.af7.slice(-FRAME_SAMPLES),
        af8:  buf.af8.slice(-FRAME_SAMPLES),
        tp10: buf.tp10.slice(-FRAME_SAMPLES),
      },
    }
    try {
      await fetch(`${apiUrl}/api/v1/neurolink/ingest`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      frameCountRef.current += 1
      setFramesSent(frameCountRef.current)
    } catch { /* network blip — drop frame, keep streaming */ }
  }, [apiUrl])

  // Wire up a single EEG characteristic notification
  const subscribeEEG = useCallback((char: BluetoothRemoteGATTCharacteristic, channel: keyof ChannelBuffer) => {
    char.addEventListener('characteristicvaluechanged', (evt) => {
      const val = (evt.target as BluetoothRemoteGATTCharacteristic).value
      if (!val) return
      const samples = decodeSamples(val)
      bufRef.current[channel].push(...samples)
      // Keep buffer bounded
      if (bufRef.current[channel].length > FRAME_SAMPLES * 4) {
        bufRef.current[channel] = bufRef.current[channel].slice(-FRAME_SAMPLES * 2)
      }
    })
  }, [])

  const doConnect = useCallback(async () => {
    if (!isUsable) return
    setStatus('requesting')
    setErrorMsg(null)

    let device: BluetoothDevice
    try {
      device = await (navigator as Navigator & { bluetooth: Bluetooth }).bluetooth.requestDevice({
        filters: [{ services: [MUSE_SERVICE] }],
        optionalServices: [
          CHAR_CONTROL, CHAR_EEG_TP9, CHAR_EEG_AF7,
          CHAR_EEG_AF8, CHAR_EEG_TP10, CHAR_TELEMETRY,
        ],
      })
    } catch (err) {
      // User cancelled the picker — return to idle cleanly
      setStatus('idle')
      setErrorMsg(String(err).replace('NotFoundError: ', ''))
      return
    }

    deviceRef.current = device
    setDeviceName(device.name ?? 'Muse')
    setStatus('connecting')

    // Handle device disconnection
    device.addEventListener('gattserverdisconnected', () => {
      setStatus('reconnecting')
      setContact({ tp9: false, af7: false, af8: false, tp10: false })
      if (!aliveRef.current) return
      setTimeout(() => {
        reconnDelay.current = Math.min(reconnDelay.current * 2, 30_000)
        doConnect()
      }, reconnDelay.current)
    })

    try {
      const server = await device.gatt!.connect()
      serverRef.current = server
      const service = await server.getPrimaryService(MUSE_SERVICE)

      // Subscribe to all 4 EEG channels
      const [tp9Char, af7Char, af8Char, tp10Char, telemChar, ctrlChar] = await Promise.all([
        service.getCharacteristic(CHAR_EEG_TP9),
        service.getCharacteristic(CHAR_EEG_AF7),
        service.getCharacteristic(CHAR_EEG_AF8),
        service.getCharacteristic(CHAR_EEG_TP10),
        service.getCharacteristic(CHAR_TELEMETRY),
        service.getCharacteristic(CHAR_CONTROL),
      ])

      subscribeEEG(tp9Char,  'tp9')
      subscribeEEG(af7Char,  'af7')
      subscribeEEG(af8Char,  'af8')
      subscribeEEG(tp10Char, 'tp10')

      await tp9Char.startNotifications()
      await af7Char.startNotifications()
      await af8Char.startNotifications()
      await tp10Char.startNotifications()

      // Subscribe to telemetry for battery + contact quality
      telemChar.addEventListener('characteristicvaluechanged', (evt) => {
        const val = (evt.target as BluetoothRemoteGATTCharacteristic).value
        if (!val) return
        const { battery: bat, contact: c } = decodeTelemetry(val)
        setBattery(bat)
        setContact({ tp9: c[0], af7: c[1], af8: c[2], tp10: c[3] })
      })
      await telemChar.startNotifications()

      // Send CMD_DATA to start streaming
      await ctrlChar.writeValue(CMD_DATA)

      reconnDelay.current = 2_000  // reset backoff on successful connect
      setStatus('streaming')

      // Post a frame every second
      const interval = setInterval(() => {
        if (!aliveRef.current) { clearInterval(interval); return }
        const b = bufRef.current
        if (b.tp9.length >= 5) postFrame({ ...b })
      }, 1_000)

      device.addEventListener('gattserverdisconnected', () => clearInterval(interval), { once: true })

    } catch (err) {
      setStatus('error')
      setErrorMsg(`GATT error: ${String(err)}`)
    }
  }, [isUsable, postFrame, subscribeEEG])

  const doDisconnect = useCallback(async () => {
    aliveRef.current = false
    if (serverRef.current?.connected) {
      serverRef.current.disconnect()
    }
    setStatus('idle')
    setDeviceName(null)
    setBattery(null)
    setContact({ tp9: false, af7: false, af8: false, tp10: false })
    setFramesSent(0)
    frameCountRef.current = 0
    bufRef.current = { tp9: [], af7: [], af8: [], tp10: [] }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
      if (serverRef.current?.connected) serverRef.current.disconnect()
    }
  }, [])

  return {
    status,
    deviceName,
    battery,
    contact,
    errorMsg,
    framesSent,
    connect:    doConnect,
    disconnect: doDisconnect,
  }
}
