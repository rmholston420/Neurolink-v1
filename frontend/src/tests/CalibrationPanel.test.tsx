import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import CalibrationPanel from '../components/CalibrationPanel'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const mockFetch = () => vi.mocked(fetch)

describe('CalibrationPanel', () => {
  it('renders the Start Calibration button initially', () => {
    render(<CalibrationPanel apiUrl="http://test" />)
    // Use exact button text to avoid matching the description paragraph
    expect(screen.getByRole('button', { name: 'Start Calibration' })).toBeTruthy()
  })

  it('shows loading state while request is in flight', async () => {
    mockFetch().mockReturnValue(new Promise(() => {}))
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      // Both button and status div say 'Calibrating…' — use getAllByText
      expect(screen.getAllByText(/calibrating/i).length).toBeGreaterThan(0)
    })
  })

  it('button is disabled while loading', async () => {
    mockFetch().mockReturnValue(new Promise(() => {}))
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      const btn = screen.getByRole('button')
      expect(btn.getAttribute('aria-disabled')).toBe('true')
    })
  })

  it('displays Calibration complete after successful POST', async () => {
    mockFetch().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'ok', message: 'Calibration complete', samples: 256 }),
    } as Response)
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(screen.getByText('Calibration complete')).toBeTruthy()
    })
  })

  it('displays error message after network failure', async () => {
    mockFetch().mockRejectedValue(new Error('Network error'))
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(screen.queryByText(/error|fail|Network/i)).toBeTruthy()
    })
  })

  it('displays error message when response is not ok', async () => {
    mockFetch().mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'No device connected' }),
    } as Response)
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(screen.queryByText(/error|fail|No device/i)).toBeTruthy()
    })
  })
})
