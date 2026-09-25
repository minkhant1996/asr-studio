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
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const AUDIO_EXT = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus', '.webm', '.aac', '.wma', '.aiff']
  const isAudio = (f: File) => f.type.startsWith('audio/') || AUDIO_EXT.some((e) => f.name.toLowerCase().endsWith(e))

  function accept(f: File | undefined | null) {
    if (!f) return
    if (!isAudio(f)) {
      setError(`${f.name} does not look like an audio file. Accepted: ${AUDIO_EXT.join(', ')}`)
      return
    }
    pick(f)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files?.[0]
    accept(dropped)
  }

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
        <div
          className={`dropzone ${dragging ? 'over' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragEnter={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={(e) => {
            // ignore drags moving between children of the zone
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false)
          }}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Drop an audio file here, or click to browse"
        >
          <div className="dz-icon">{dragging ? '⤓' : '🎧'}</div>
          {file ? (
            <>
              <b>{file.name}</b>
              <div className="small">
                {(file.size / 1024 / 1024).toFixed(1)} MB · click or drop another file to replace it
              </div>
            </>
          ) : (
            <>
              <b>{dragging ? 'Drop to load the audio' : 'Drag an audio file here'}</b>
              <div className="small">or click to browse · mp3, wav, m4a, ogg, flac, opus · up to 200 MB</div>
            </>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.opus,.webm,.aac"
          style={{ display: 'none' }}
          onChange={(e) => accept(e.target.files?.[0])}
        />
        <div className="small" style={{ marginTop: 6 }}>Files longer than 30 seconds are split into chunks automatically.</div>
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
