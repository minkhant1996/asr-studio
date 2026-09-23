import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import DataTab from './components/DataTab'
import EvaluateTab from './components/EvaluateTab'
import SettingsTab from './components/SettingsTab'
import SystemStats from './components/SystemStats'
import TrainTab from './components/TrainTab'
import TranscribeTab from './components/TranscribeTab'
import type { EvalRecord, Manifest, RunSummary } from './types'

type Tab = 'data' | 'train' | 'evaluate' | 'transcribe' | 'settings'

export default function App() {
  const [tab, setTab] = useState<Tab>('data')
  const [health, setHealth] = useState<{ ok: boolean; device: string; hf_token_set: boolean } | null>(null)
  const [prepared, setPrepared] = useState<Manifest[]>([])
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [evals, setEvals] = useState<EvalRecord[]>([])

  const loadHealth = useCallback(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])
  const loadPrepared = useCallback(() => {
    api.prepared().then(setPrepared).catch(() => setPrepared([]))
  }, [])
  const loadRuns = useCallback(() => {
    api.runs().then(setRuns).catch(() => setRuns([]))
  }, [])
  const loadEvals = useCallback(() => {
    api.evals().then(setEvals).catch(() => setEvals([]))
  }, [])

  useEffect(() => {
    loadHealth()
    loadPrepared()
    loadRuns()
    loadEvals()
  }, [loadHealth, loadPrepared, loadRuns, loadEvals])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'data', label: '1 · Data' },
    { id: 'train', label: '2 · Fine-tune' },
    { id: 'evaluate', label: '3 · Evaluate' },
    { id: 'transcribe', label: '4 · Transcribe' },
    { id: 'settings', label: 'Settings' },
  ]

  return (
    <div className="app">
      <header>
        <h1>ASR Studio</h1>
        <span className="status">
          {health ? `backend ok · ${health.device.toUpperCase()}` : 'backend offline'}
          {health && <SystemStats compact />}
        </span>
      </header>
      <nav className="tabs">
        {tabs.map((t) => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      {tab === 'data' && <DataTab prepared={prepared} reload={loadPrepared} onPrepared={() => loadPrepared()} />}
      {tab === 'train' && <TrainTab prepared={prepared} runs={runs} reloadRuns={loadRuns} />}
      {tab === 'evaluate' && <EvaluateTab prepared={prepared} runs={runs} evals={evals} reloadEvals={loadEvals} />}
      {tab === 'transcribe' && <TranscribeTab runs={runs} />}
      {tab === 'settings' && <SettingsTab onChange={loadHealth} />}
    </div>
  )
}
