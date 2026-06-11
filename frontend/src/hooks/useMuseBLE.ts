/**
 * useMuseBLE — Path A: Web Bluetooth connection to Muse S Gen 1.
 *
 * Uses the Web Bluetooth API (Chrome/Edge/Opera only) to:
 *   1. Let the user pick their Muse from the browser’s native BLE picker.
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
 * Reconnect note:
 *   requestDevice() requires a user gesture — calling it from a timer throws
 *   SecurityError. On reconnect we already hold a BluetoothDevice reference
 *   so we call device.gatt.connect() directly and re-subscribe without going
 *   through the picker again.
 *
 * optionalServices note:
 *   filters: [{ services: [MUSE_SERVICE] }] grants access to all
 *   characteristics under 0xfe8d. optionalServices is omitted — it expects
 *   service UUIDs, not characteristic UUIDs.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

// ── GATT UUIDs ────────────────────────────────────────────────────────────────
const MUSE_SERVICE   = 0xfe8d
const CHAR_CONTROL   = '273e0001-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_TP9   = '273e0003-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_AF7   = '273e0004-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_AF8   = '273e0005-4c4d-454d-96be-f03bac821358'
const CHAR_EEG_TP10  = '273e0006-4c4d-454d-96be-f03bac821358'
const CHAR_TELEMETRY = '273e000b-4c4d-454d-96be-f03bac821358'

// CMD_DATA: tell the Muse to start streaming EEG + telemetry
const CMD_DATA = new Uint8Array([0x02, 0x64, 0x0a])

// ── Packet decode ─────────────────────────────────────────────────────────────
function decodeSamples(buf: DataView): number[] {
  const out: number[] = []
  for (let i = 0; i < 5; i++) {
    const byteOffset = 2 + Math.floor(i * 12 / 8)
    const bitShift   = (i % 2 === 0) ? 4 : 0
    const hi = buf.getUint8(byteOffset)
    const lo = buf.getUint8(byteOffset + 1)
    const raw = ((hi << 8 | lo) >> bitShift) & 0xfff
    out.push((raw - 2048) * 0.48828125)
  }
  return out
}

// ── Telemetry decode ──────────────────────────────────────────────────────────
function decodeTelemetry(buf: DataView): { battery: number; contact: boolean[] } {
  const battery = buf.byteLength > 1 ? Math.round((buf.getUint8(1) >> 1) * 100 / 63) : 0
  const contactByte = buf.byteLength > 5 ? buf.getUint8(5) : 0
  const contact = [
    (contactByte & 0x10) === 0,
    (contactByte & 0x20) === 0,
    (contactByte & 0x40) === 0,
    (contactByte & 0x80) === 0,
  ]
  return { battery, contact }
}

// ── Types ─────────────────────────────────────────────────────────────────────
export type BLEStatus =
  | 'unsupported'
  | 'insecure_origin'
  | 'idle'
  | 'requesting'
  | 'connecting'
  | 'streaming'
  | 'reconnecting'
  | 'error'

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
function detectBLEStatus(): BLEStatus {
  if (typeof window === 'undefined') return 'unsupported'
  if (!window.isSecureContext) return 'insecure_origin'
  if (typeof navigator === 'undefined' || !('bluetooth' in navigator)) return 'unsupported'
  return 'idle'
}

// ── Frame accumulator ─────────────────────────────────────────────────────────
const FRAME_SAMPLES = 256

interface ChannelBuffer {
  tp9:  number[]
  af7:  number[]
  af8:  number[]
  tp10: number[]
}

function rms(arr: number[]): number {
  if (arr.length === 0) return 0
  return Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length)
}

function estimateBandPowers(samples: number[]) {
  const r = rms(samples)
  return {
    delta: r * 0.35,
    theta: r * 0.25,
    alpha: r * 0.20,
    beta:  r * 0.15,
    gamma: r * 0.05,
  }
}

// ── GATT session setup (shared by first connect + reconnect) ────────────────────
// Accepts an already-open BluetoothRemoteGATTServer and wires up all
// characteristics. Returns a cleanup function that clears the polling interval.
async function setupGATTSession(
  server: BluetoothRemoteGATTServer,
  bufRef: React.MutableRefObject<ChannelBuffer>,
  aliveRef: React.MutableRefObject<boolean>,
  setBattery: (v: number) => void,
  setContact: (v: ContactQuality) => void,
  postFrame: (buf: ChannelBuffer) => Promise<void>,
): Promise<() => void> {
  const service = await server.getPrimaryService(MUSE_SERVICE)

  const [tp9Char, af7Char, af8Char, tp10Char, telemChar, ctrlChar] = await Promise.all([
    service.getCharacteristic(CHAR_EEG_TP9),
    service.getCharacteristic(CHAR_EEG_AF7),
    service.getCharacteristic(CHAR_EEG_AF8),
    service.getCharacteristic(CHAR_EEG_TP10),
    service.getCharacteristic(CHAR_TELEMETRY),
    service.getCharacteristic(CHAR_CONTROL),
  ])

  const subscribeEEG = (char: BluetoothRemoteGATTCharacteristic, channel: keyof ChannelBuffer) => {
    char.addEventListener('characteristicvaluechanged', (evt) => {
      const val = (evt.target as BluetoothRemoteGATTCharacteristic).value
      if (!val) return
      bufRef.current[channel].push(...decodeSamples(val))
      if (bufRef.current[channel].length > FRAME_SAMPLES * 4) {
        bufRef.current[channel] = bufRef.current[channel].slice(-FRAME_SAMPLES * 2)
      }
    })
  }

  subscribeEEG(tp9Char,  'tp9')
  subscribeEEG(af7Char,  'af7')
  subscribeEEG(af8Char,  'af8')
  subscribeEEG(tp10Char, 'tp10')

  await tp9Char.startNotifications()
  await af7Char.startNotifications()
  await af8Char.startNotifications()
  await tp10Char.startNotifications()

  telemChar.addEventListener('characteristicvaluechanged', (evt) => {
    const val = (evt.target as BluetoothRemoteGATTCharacteristic).value
    if (!val) return
    const { battery: bat, contact: c } = decodeTelemetry(val)
    setBattery(bat)
    setContact({ tp9: c[0], af7: c[1], af8: c[2], tp10: c[3] })
  })
  await telemChar.startNotifications()

  await ctrlChar.writeValue(CMD_DATA)

  const interval = setInterval(() => {
    if (!aliveRef.current) { clearInterval(interval); return }
    const b = bufRef.current
    if (b.tp9.length >= 5) postFrame({ ...b })
  }, 1_000)

  return () => clearInterval(interval)
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

  const deviceRef     = useRef<BluetoothDevice | null>(null)
  const serverRef     = useRef<BluetoothRemoteGATTServer | null>(null)
  const aliveRef      = useRef(true)
  const reconnDelay   = useRef(2_000)
  const cleanupRef    = useRef<(() => void) | null>(null)
  const bufRef        = useRef<ChannelBuffer>({ tp9: [], af7: [], af8: [], tp10: [] })
  const frameCountRef = useRef(0)

  const postFrame = useCallback(async (buf: ChannelBuffer) => {
    const allSamples = [
      ...buf.tp9.slice(-FRAME_SAMPLES),
      ...buf.af7.slice(-FRAME_SAMPLES),
      ...buf.af8.slice(-FRAME_SAMPLES),
      ...buf.tp10.slice(-FRAME_SAMPLES),
    ]
    const payload = {
      source: 'ble_webbt',
      ts: Date.now() / 1000,
      bands: estimateBandPowers(allSamples),
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
    } catch { /* network blip — drop frame */ }
  }, [apiUrl])

  // Reconnect using the existing device ref — no user gesture needed.
  // Never calls requestDevice() again.
  const doReconnect = useCallback(async () => {
    const device = deviceRef.current
    if (!device || !aliveRef.current) return

    try {
      cleanupRef.current?.()
      const server = await device.gatt!.connect()
      serverRef.current = server
      const cleanup = await setupGATTSession(
        server, bufRef, aliveRef,
        (v) => setBattery(v),
        (v) => setContact(v),
        postFrame,
      )
      cleanupRef.current = cleanup
      reconnDelay.current = 2_000
      setStatus('streaming')
    } catch (err) {
      if (!aliveRef.current) return
      reconnDelay.current = Math.min(reconnDelay.current * 2, 30_000)
      setTimeout(doReconnect, reconnDelay.current)
    }
  }, [postFrame])

  const doConnect = useCallback(async () => {
    if (!isUsable) return
    setStatus('requesting')
    setErrorMsg(null)

    let device: BluetoothDevice
    try {
      device = await (navigator as Navigator & { bluetooth: Bluetooth }).bluetooth.requestDevice({
        // filters grants access to all characteristics under 0xfe8d.
        // optionalServices omitted — it expects service UUIDs, not char UUIDs.
        filters: [{ services: [MUSE_SERVICE] }],
      })
    } catch (err) {
      setStatus('idle')
      setErrorMsg(String(err).replace('NotFoundError: ', ''))
      return
    }

    deviceRef.current = device
    setDeviceName(device.name ?? 'Muse')
    setStatus('connecting')

    // On disconnect: update UI and retry via doReconnect (no picker needed)
    device.addEventListener('gattserverdisconnected', () => {
      cleanupRef.current?.()
      setStatus('reconnecting')
      setContact({ tp9: false, af7: false, af8: false, tp10: false })
      if (!aliveRef.current) return
      setTimeout(() => {
        reconnDelay.current = Math.min(reconnDelay.current * 2, 30_000)
        doReconnect()
      }, reconnDelay.current)
    })

    try {
      const server = await device.gatt!.connect()
      serverRef.current = server
      const cleanup = await setupGATTSession(
        server, bufRef, aliveRef,
        (v) => setBattery(v),
        (v) => setContact(v),
        postFrame,
      )
      cleanupRef.current = cleanup
      reconnDelay.current = 2_000
      setStatus('streaming')
    } catch (err) {
      setStatus('error')
      setErrorMsg(`GATT error: ${String(err)}`)
    }
  }, [isUsable, postFrame, doReconnect])

  const doDisconnect = useCallback(async () => {
    aliveRef.current = false
    cleanupRef.current?.()
    if (serverRef.current?.connected) serverRef.current.disconnect()
    setStatus('idle')
    setDeviceName(null)
    setBattery(null)
    setContact({ tp9: false, af7: false, af8: false, tp10: false })
    setFramesSent(0)
    frameCountRef.current = 0
    bufRef.current = { tp9: [], af7: [], af8: [], tp10: [] }
  }, [])

  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
      cleanupRef.current?.()
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
