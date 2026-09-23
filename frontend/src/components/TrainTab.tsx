import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { Manifest, ModelSpec, RunSummary } from '../types'
import { useConfirm } from './ConfirmDialog'
import LossChart from './LossChart'

export default function TrainTab({ prepared, runs, reloadRuns }: { prepared: Manifest[]; runs: RunSummary[]; reloadRuns: () => void }) {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [hw, setHw] = useState<{ device: string; gpu_gb: number; free_ram_gb: number } | null>(null)
  const [datasetId, setDatasetId] = useState('')
  const [model, setModel] = useState('whisper-small')
  const [method, setMethod] = useState<'lora' | 'full'>('lora')
  const [maxSteps, setMaxSteps] = useState(100)
  const [batch, setBatch] = useState(2)
  const [accum, setAccum] = useState(4)
  const [lr, setLr] = useState(1e-3)
  const [warmup, setWarmup] = useState(10)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [live, setLive] = useState<any>(null)
  const [losses, setLosses] = useState<{ step: number; loss: number }[]>([])
  const [runId, setRunId] = useState('')
  const [error, setError] = useState('')
  const [viewing, setViewing] = useState<RunSummary | null>(null)
  const [histOpen, setHistOpen] = useState(true)
  const abortRef = useRef<AbortController | null>(null)
  const { confirm, dialog } = useConfirm()

  useEffect(() => {
    api.models().then((r) => {
      setModels(r.models)
      setHw({ device: r.device, gpu_gb: r.gpu_gb, free_ram_gb: r.free_ram_gb })
    }).catch(() => setModels([]))
  }, [])
  useEffect(() => {
    if (!datasetId && prepared.length) setDatasetId(prepared[0].id)
  }, [prepared, datasetId])

  const spec = models.find((m) => m.id === model)
  const ds = prepared.find((p) => p.id === datasetId)
  const fits = spec ? (method === 'lora' ? spec.fits_lora : spec.fits_full) : true
  const canTrain = !!spec?.can_train
  const stepsPerEpoch = ds ? Math.max(1, Math.floor(ds.train / Math.max(1, batch * accum))) : 0
  const ready = !!datasetId && canTrain

  async function run() {
    if (!ready) return
    setError('')
    setBusy(true)
    setLosses([])
    setLive(null)
    setRunId('')
    setViewing(null)
    setStatus('starting')
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.trainStream(
        { dataset_id: datasetId, model, method, max_steps: maxSteps, batch_size: batch, grad_accum: accum,
          learning_rate: lr, warmup_steps: warmup, name, fp16: hw?.device === 'cuda' },
        (ev) => {
          if (ev.type === 'run') setRunId(ev.run.id)
          else if (ev.type === 'status') setStatus(ev.message)
          else if (ev.type === 'start') { setStatus(`training on ${ev.device}`); setLive({ ...ev, step: 0 }) }
          else if (ev.type === 'step') {
            setLive((p: any) => ({ ...(p || {}), ...ev }))
            if (ev.loss != null) setLosses((l) => [...l, { step: ev.step, loss: ev.loss }])
          } else if (ev.type === 'done') { setStatus(ev.stopped_early ? 'stopped early' : 'done'); reloadRuns() }
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
      reloadRuns()
    }
  }

  const shown = viewing ?? (runId ? ({ id: runId, losses } as any) : null)
  const chartPoints = viewing ? viewing.losses ?? [] : losses

  const historyPanel = histOpen ? (
    <section className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>Runs ({runs.length})</h2>
        <button className="iconbtn" onClick={() => setHistOpen(false)} title="Minimize">☰</button>
      </div>
      <div className="sessions" style={{ maxHeight: '70vh', marginTop: 8 }}>
        {runs.length === 0 && <div className="small">Training runs are saved here with their loss curve and checkpoint.</div>}
        {runs.map((r) => (
          <div key={r.id} className={`session ${viewing?.id === r.id ? 'on' : ''}`} onClick={() => api.runGet(r.id).then(setViewing)} title={`${r.name} · ${new Date(r.created * 1000).toLocaleString()}`}>
            <span>
              <span className={`kind ${r.status === 'error' ? 'bad' : ''}`}>{r.status}</span> {r.name}
            </span>
            <small>{r.steps ?? '—'}</small>
            <button
              title="Delete"
              onClick={async (e) => {
                e.stopPropagation()
                if (await confirm({ title: 'Delete this run?', message: <span>“{r.name}” and its checkpoint will be removed.</span> })) {
                  await api.runDelete(r.id)
                  if (viewing?.id === r.id) setViewing(null)
                  reloadRuns()
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
      <button className="iconbtn" onClick={() => setHistOpen(true)} title="Show runs">☰</button>
      <span className="count">Runs ({runs.length})</span>
    </section>
  )

  return (
    <div className={`wrap2 ${histOpen ? '' : 'min'}`}>
      {dialog}
      {historyPanel}
      <div>
        <section className="panel">
          <h2>Fine-tune</h2>
          <div className="row" style={{ marginTop: 0 }}>
            <label>dataset</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} style={{ maxWidth: 380 }}>
              <option value="">—</option>
              {prepared.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.clips} clips ({p.train} train / {p.test} test)
                </option>
              ))}
            </select>
            {prepared.length === 0 && <span className="small">Prepare a dataset in the Data tab first.</span>}
          </div>

          <div className="small" style={{ margin: '10px 0 4px' }}>base model</div>
          <div className="cards">
            {models.map((m) => (
              <div key={m.id} className={`card ${model === m.id ? 'on' : ''}`} onClick={() => setModel(m.id)} title={m.note}>
                <b>{m.label}</b>
                <div className="small">{m.note}</div>
                <div className="meta">
                  <span className="tag">{m.params}</span>
                  {m.can_train ? <span className="tag ok">fine-tunable</span> : <span className="tag warn">inference only</span>}
                  {hw?.device === 'cuda'
                    ? <span className={`tag ${m.fits_lora ? 'ok' : 'warn'}`}>LoRA ~{m.vram_lora_gb} GB VRAM</span>
                    : <span className={`tag ${m.fits_lora ? '' : 'warn'}`}>~{m.ram_cpu_gb} GB RAM</span>}
                </div>
              </div>
            ))}
          </div>

          <div className="row">
            <label>method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value as 'lora' | 'full')} style={{ width: 210 }}>
              <option value="lora">LoRA adapters (light)</option>
              <option value="full" disabled={spec?.finetune === 'lora'}>full fine-tune</option>
            </select>
            <label>steps</label>
            <input type="number" min={1} max={100000} value={maxSteps} onChange={(e) => setMaxSteps(+e.target.value)} style={{ width: 90 }} />
            <label>batch</label>
            <input type="number" min={1} max={64} value={batch} onChange={(e) => setBatch(+e.target.value)} style={{ width: 70 }} />
            <label title="Effective batch = batch × accumulation">× accum</label>
            <input type="number" min={1} max={64} value={accum} onChange={(e) => setAccum(+e.target.value)} style={{ width: 70 }} />
            <label>learning rate</label>
            <input type="number" step={1e-5} value={lr} onChange={(e) => setLr(+e.target.value)} style={{ width: 110 }} />
            <label>warmup</label>
            <input type="number" min={0} value={warmup} onChange={(e) => setWarmup(+e.target.value)} style={{ width: 80 }} />
          </div>
          <div className="row">
            <label>run name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="optional" style={{ width: 240 }} />
            <span className="small">
              effective batch {batch * accum}
              {ds ? ` · ${stepsPerEpoch} steps = 1 epoch over ${ds.train} clips` : ''}
            </span>
          </div>
          <div className="row">
            <button className="primary" disabled={busy || !ready} onClick={run} title={!datasetId ? 'Pick a prepared dataset' : !canTrain ? `${spec?.label} cannot be fine-tuned here` : ''}>
              {busy ? 'Training…' : 'Start fine-tuning'}
            </button>
            {busy && (
              <>
                <button className="ghost" onClick={() => runId && api.runStop(runId)}>Stop after this step</button>
                <button className="ghost" onClick={() => abortRef.current?.abort()}>Abort</button>
              </>
            )}
            {!canTrain && spec && <span className="small" style={{ color: 'var(--bad)' }}>{spec.label} is inference only here. Use it in Evaluate and Transcribe.</span>}
            {canTrain && !fits && <span className="small" style={{ color: '#ffb454' }}>⚠ This may not fit in memory on this machine.</span>}
          </div>
          {error && <div className="error">{error}</div>}
        </section>

        {(busy || live || chartPoints.length > 0 || viewing) && (
          <section className="panel" style={{ marginTop: 16 }}>
            <h2>{viewing ? `Run · ${viewing.name}` : 'Progress'}</h2>
            {!viewing && (
              <>
                <div className="live">
                  <span>status: <b>{status}</b></span>
                  {live?.max_steps && <span>step <b>{live.step ?? 0}</b> / {live.max_steps}</span>}
                  {live?.loss != null && <span>loss <b>{Number(live.loss).toFixed(4)}</b></span>}
                  {live?.elapsed != null && <span>elapsed <b>{live.elapsed}s</b></span>}
                  {live?.eta != null && <span>ETA <b>{live.eta}s</b></span>}
                  {live?.ram_mb != null && <span>RAM <b>{(live.ram_mb / 1024).toFixed(2)} GB</b></span>}
                  {live?.vram_mb != null && <span>VRAM <b>{(live.vram_mb / 1024).toFixed(2)} GB</b></span>}
                </div>
                <div className="progress">
                  <div style={{ width: live?.max_steps ? `${((live.step ?? 0) / live.max_steps) * 100}%` : '0%' }} />
                </div>
              </>
            )}
            {viewing && (
              <div className="metrics" style={{ flexWrap: 'wrap' }}>
                <div className="metric"><div className="k">status</div><div className="v" style={{ fontSize: 16 }}>{viewing.status}</div></div>
                <div className="metric"><div className="k">steps</div><div className="v">{viewing.steps ?? '—'}</div></div>
                <div className="metric"><div className="k">final loss</div><div className="v">{viewing.train_loss != null ? viewing.train_loss.toFixed(4) : '—'}</div></div>
                <div className="metric"><div className="k">runtime</div><div className="v" style={{ fontSize: 18 }}>{viewing.train_runtime ?? '—'}s</div></div>
                <div className="metric"><div className="k">base model</div><div className="v" style={{ fontSize: 15 }}>{viewing.model_label}</div></div>
              </div>
            )}
            {viewing?.message && <div className="error">{viewing.message}</div>}
            <div style={{ marginTop: 10 }}>
              <LossChart points={chartPoints} />
            </div>
            {shown && !viewing && runId && <div className="small">Run id {runId}. It appears in Evaluate as a fine-tuned target once finished.</div>}
          </section>
        )}
      </div>
    </div>
  )
}
