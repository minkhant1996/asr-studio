import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { ModelSpec, RunSummary, Target } from '../types'

export default function TranscribeTab({ runs }: { runs: RunSummary[] }) {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [target, setTarget] = useState<Target>({ kind: 'base', model: 'whisper-small' })
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.transcribe>> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.models().then((r) => setModels(r.models)).catch(() => setModels([]))
  }, [])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])

  const doneRuns = runs.filter((r) => r.status === 'done' && r.model_dir)
  const key = target.kind === 'base' ? `base:${target.model}` : `run:${target.run_id}`

  function pick(f: File | null) {
    setResult(null)
    setError('')
    setFile(f)
    if (url) URL.revokeObjectURL(url)
    setUrl(f ? URL.createObjectURL(f) : '')
  }

  async function run() {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)
    try {
      setResult(await api.transcribe(file, target))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid">
      <section className="panel">
        <h2>Transcribe an audio file</h2>
        <input ref={inputRef} type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.opus,.webm" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        <div className="small" style={{ marginTop: 6 }}>mp3, wav, m4a, ogg, flac, opus. Long files are split into 30-second chunks. Max 200 MB.</div>
        {url && <audio controls src={url} />}
        <div className="row">
          <label>model</label>
          <select
            value={key}
            onChange={(e) => {
              const [kind, v] = e.target.value.split(/:(.+)/)
              setTarget(kind === 'base' ? { kind: 'base', model: v } : { kind: 'run', run_id: v })
            }}
            style={{ maxWidth: 380 }}
          >
            <optgroup label="base models">
              {models.map((m) => (
                <option key={m.id} value={`base:${m.id}`}>{m.label} ({m.params})</option>
              ))}
            </optgroup>
            <optgroup label="fine-tuned runs">
              {doneRuns.map((r) => (
                <option key={r.id} value={`run:${r.id}`}>{r.name}</option>
              ))}
            </optgroup>
          </select>
          <button className="primary" disabled={!file || busy} onClick={run}>{busy ? 'Transcribing…' : 'Transcribe'}</button>
        </div>
        {busy && <div className="small">First use of a model downloads and loads it, which can take a minute.</div>}
        {error && <div className="error">{error}</div>}
      </section>
      <section className="panel">
        <h2>Transcript</h2>
        {!result && <div className="small">The transcript appears here, in Burmese script when the model supports it.</div>}
        {result && (
          <>
            <div className="metrics" style={{ flexWrap: 'wrap' }}>
              <div className="metric"><div className="k">audio</div><div className="v" style={{ fontSize: 18 }}>{result.duration}s</div></div>
              <div className="metric"><div className="k">took</div><div className="v" style={{ fontSize: 18 }}>{(result.ms / 1000).toFixed(2)}s</div></div>
              <div className="metric"><div className="k">real-time factor</div><div className="v" style={{ fontSize: 18 }}>{result.rtf ?? '—'}</div></div>
              <div className="metric"><div className="k">device</div><div className="v" style={{ fontSize: 16 }}>{result.device}</div></div>
            </div>
            <div className="clip">
              <div className="my" style={{ fontSize: 15 }}>{result.text || <span className="small">(empty)</span>}</div>
            </div>
            <div className="small">{result.model} · {result.chunks.length} chunk{result.chunks.length === 1 ? '' : 's'}</div>
            <div className="row">
              <button className="ghost" onClick={() => navigator.clipboard?.writeText(result.text)}>Copy transcript</button>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
