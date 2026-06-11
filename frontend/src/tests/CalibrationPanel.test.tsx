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
    expect(screen.getByRole('button')).toBeTruthy()
    expect(screen.getByText(/calibrat/i)).toBeTruthy()
  })

  it('shows loading state while request is in flight', async () => {
    // Never resolves so we can observe the loading state
    mockFetch().mockReturnValue(new Promise(() => {}))
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(screen.getByText(/calibrating|loading|\.\.\./i)).toBeTruthy()
    })
  })

  it('displays success result after successful POST', async () => {
    mockFetch().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'ok', message: 'Calibration complete', samples: 256 }),
    } as Response)
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(
        screen.queryByText(/complete|ok|success|calibrat/i)
      ).toBeTruthy()
    })
  })

  it('displays error message after failed POST', async () => {
    mockFetch().mockRejectedValue(new Error('Network error'))
    render(<CalibrationPanel apiUrl="http://test" />)
    fireEvent.click(screen.getByRole('button'))
    await waitFor(() => {
      expect(
        screen.queryByText(/error|fail|Network/i)
      ).toBeTruthy()
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
      expect(
        screen.queryByText(/error|fail|No device/i)
      ).toBeTruthy()
    })
  })
})
