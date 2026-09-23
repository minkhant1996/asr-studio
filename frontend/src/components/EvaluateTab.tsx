import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { EvalRecord, EvalTargetResult, Manifest, ModelSpec, RunSummary, Target } from '../types'
import { useConfirm } from './ConfirmDialog'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export default function EvaluateTab({ prepared, runs, evals, reloadEvals }: { prepared: Manifest[]; runs: RunSummary[]; evals: EvalRecord[]; reloadEvals: () => void }) {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [split, setSplit] = useState('test')
  const [limit, setLimit] = useState(20)
  const [targets, setTargets] = useState<Target[]>([{ kind: 'base', model: 'whisper-small' }])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [live, setLive] = useState<any>(null)
  const [partial, setPartial] = useState<EvalTargetResult[]>([])
  const [record, setRecord] = useState<EvalRecord | null>(null)
  const [error, setError] = useState('')
  const [histOpen, setHistOpen] = useState(true)
  const abortRef = useRef<AbortController | null>(null)
  const { confirm, dialog } = useConfirm()

  useEffect(() => {
    api.models().then((r) => setModels(r.models)).catch(() => setModels([]))
  }, [])
  useEffect(() => {
    if (!datasetId && prepared.length) setDatasetId(prepared[0].id)
  }, [prepared, datasetId])

  const ds = prepared.find((p) => p.id === datasetId)
  const maxN = ds ? (split === 'test' ? ds.test : ds.train) : 0
  const doneRuns = runs.filter((r) => r.status === 'done' && r.model_dir)

  const targetKey = (t: Target) => (t.kind === 'base' ? `base:${t.model}` : `run:${t.run_id}`)
  function setTarget(i: number, key: string) {
    const [kind, v] = key.split(/:(.+)/)
    setTargets(targets.map((t, j) => (j === i ? (kind === 'base' ? { kind: 'base', model: v } : { kind: 'run', run_id: v }) : t)))
  }

  async function run() {
    if (!datasetId) return
    setError('')
    setBusy(true)
    setRecord(null)
    setPartial([])
    setLive(null)
    setStatus('starting')
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.evaluateStream(
        { dataset_id: datasetId, targets, split, limit },
        (ev) => {
          if (ev.type === 'status') setStatus(ev.message)
          else if (ev.type === 'progress') setLive(ev)
          else if (ev.type === 'target_done') setPartial((p) => [...p, ev.result])
          else if (ev.type === 'done') { setRecord(ev.evaluation); setStatus('done'); reloadEvals() }
          else if (ev.type === 'error') { setError(ev.message); setStatus('failed') }
        },
        ac.signal,
      )
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
      else setStatus('stopped')
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  const results = record?.results ?? partial
  const best = results.length ? results.reduce((a, b) => (a.cer <= b.cer ? a : b)) : null

  const historyPanel = histOpen ? (
    <section className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>Evaluations ({evals.length})</h2>
        <button className="iconbtn" onClick={() => setHistOpen(false)} title="Minimize">☰</button>
      </div>
      <div className="sessions" style={{ maxHeight: '70vh', marginTop: 8 }}>
        {evals.length === 0 && <div className="small">Every evaluation is saved here with its per-clip transcripts.</div>}
        {evals.map((e) => (
          <div key={e.id} className={`session ${record?.id === e.id ? 'on' : ''}`} onClick={() => api.evalGet(e.id).then((r) => { setRecord(r); setPartial([]) })} title={new Date(e.created * 1000).toLocaleString()}>
            <span>{e.title}</span>
            <small>{e.summary?.[0] ? pct(e.summary[0].cer) : ''}</small>
            <button
              title="Delete"
              onClick={async (ev) => {
                ev.stopPropagation()
                if (await confirm({ title: 'Delete this evaluation?', message: <span>“{e.title}” will be removed.</span> })) {
                  await api.evalDelete(e.id)
                  if (record?.id === e.id) setRecord(null)
                  reloadEvals()
                }
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </section>
  ) : (
    <section className="panel hist-min">
      <button className="iconbtn" onClick={() => setHistOpen(true)} title="Show evaluations">☰</button>
      <span className="count">Evals ({evals.length})</span>
    </section>
  )

  return (
    <div className={`wrap2 ${histOpen ? '' : 'min'}`}>
      {dialog}
      {historyPanel}
      <div>
        <section className="panel">
          <h2>Measure word and character error rate</h2>
          <div className="row" style={{ marginTop: 0 }}>
            <label>dataset</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} style={{ maxWidth: 340 }}>
              <option value="">—</option>
              {prepared.map((p) => (
                <option key={p.id} value={p.id}>{p.name} · {p.clips} clips</option>
              ))}
            </select>
            <label>split</label>
            <select value={split} onChange={(e) => setSplit(e.target.value)} style={{ width: 130 }}>
              <option value="test">test{ds ? ` (${ds.test})` : ''}</option>
              <option value="train">train{ds ? ` (${ds.train})` : ''}</option>
            </select>
            <label>clips</label>
            <input type="number" min={1} max={Math.max(1, maxN)} value={limit} onChange={(e) => setLimit(Math.min(Math.max(1, +e.target.value || 1), Math.max(1, maxN)))} style={{ width: 90 }} />
            <span className="small">of {maxN}</span>
          </div>

          <div className="small" style={{ margin: '10px 0 4px' }}>models to compare</div>
          {targets.map((t, i) => (
            <div className="row" key={i} style={{ marginTop: 4 }}>
              <select value={targetKey(t)} onChange={(e) => setTarget(i, e.target.value)} style={{ maxWidth: 420 }}>
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
              {targets.length > 1 && <button className="iconbtn" onClick={() => setTargets(targets.filter((_, j) => j !== i))}>✕</button>}
            </div>
          ))}
          <div className="row">
            <button className="chip" disabled={targets.length >= 4} onClick={() => setTargets([...targets, { kind: 'base', model: 'whisper-tiny' }])}>+ add a model</button>
            <button className="primary" disabled={busy || !datasetId} onClick={run}>{busy ? 'Evaluating…' : 'Evaluate'}</button>
            {busy && <button className="ghost" onClick={() => abortRef.current?.abort()}>Stop</button>}
            {doneRuns.length === 0 && <span className="small">Fine-tune a model to compare it against its base here.</span>}
          </div>
          {(busy || live) && (
            <div style={{ marginTop: 10 }}>
              <div className="live">
                <span>status: <b>{status}</b></span>
                {live && (
                  <>
                    <span>{live.label}</span>
                    <span>clip <b>{live.i}</b>/{live.n}</span>
                    <span>CER <b>{pct(live.cer)}</b></span>
                    <span>WER <b>{pct(live.wer)}</b></span>
                    <span>avg <b>{live.avg_ms} ms</b></span>
                    <span>ETA <b>{live.eta}s</b></span>
                  </>
                )}
              </div>
              <div className="progress"><div style={{ width: live ? `${(live.i / live.n) * 100}%` : '0%' }} /></div>
            </div>
          )}
          {error && <div className="error">{error}</div>}
        </section>

        {results.length > 0 && (
          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Results{record ? ` · ${record.n} clips of ${record.dataset_name}` : ''}</h2>
            <div className="metrics" style={{ flexWrap: 'wrap' }}>
              {results.map((r) => (
                <div className="metric" key={r.label} style={{ minWidth: 190 }}>
                  <div className="k">{r.label}</div>
                  <div className="hero" style={{ color: best && r.label === best.label ? 'var(--ok)' : undefined }}>
                    {pct(r.cer)} <small>CER</small>
                  </div>
                  <div className="small">WER {pct(r.wer)} · {r.avg_ms} ms/clip{r.rtf != null ? ` · RTF ${r.rtf}` : ''}</div>
                </div>
              ))}
            </div>
            <div className="small" style={{ marginBottom: 10 }}>
              Burmese is written without spaces between most words, so <b>character error rate is the metric to trust</b>; word error
              rate depends on how the reference happens to be segmented. Lower is better; 100% CER means the output shares nothing with the reference.
            </div>
            <table className="aligned">
              <thead>
                <tr>
                  <th>model</th>
                  <th style={{ textAlign: 'right' }}>CER</th>
                  <th style={{ textAlign: 'right' }}>WER</th>
                  <th style={{ textAlign: 'right' }}>ms / clip</th>
                  <th style={{ textAlign: 'right' }}>RTF</th>
                  <th style={{ textAlign: 'right' }}>device</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.label}>
                    <td>{r.label}</td>
                    <td style={{ textAlign: 'right' }} className={best && r.label === best.label ? 'ok' : ''}>{pct(r.cer)}</td>
                    <td style={{ textAlign: 'right' }}>{pct(r.wer)}</td>
                    <td style={{ textAlign: 'right' }} className="num">{r.avg_ms}</td>
                    <td style={{ textAlign: 'right' }} className="num">{r.rtf ?? '—'}</td>
                    <td style={{ textAlign: 'right' }} className="small">{r.device}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {record?.results?.[0]?.rows && (
              <details open>
                <summary>Per-clip transcripts</summary>
                <div className="scroll" style={{ maxHeight: 460 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>reference</th>
                        {record.results.map((r) => <th key={r.label}>{r.label}</th>)}
                        <th style={{ width: 70, textAlign: 'right' }}>sec</th>
                      </tr>
                    </thead>
                    <tbody>
                      {record.results[0].rows!.map((row, i) => (
                        <tr key={i}>
                          <td className="my">{row.reference}</td>
                          {record.results.map((r) => {
                            const h = r.rows?.[i]
                            return (
                              <td key={r.label} className="my">
                                {h?.hypothesis || <span className="small">(empty)</span>}
                                <div className="small">CER {h ? pct(h.cer) : '—'}</div>
                              </td>
                            )
                          })}
                          <td className="num small" style={{ textAlign: 'right' }}>{row.duration}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
