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
  const [evals, setEvals] = useState<{ step: number; cer: number; wer: number; n: number }[]>([])
  const [lastEval, setLastEval] = useState<any>(null)
  const [evalEvery, setEvalEvery] = useState(0)
  const [evalClips, setEvalClips] = useState(8)
  const [language, setLanguage] = useState('')
  const [earlyStop, setEarlyStop] = useState(true)
  const [patience, setPatience] = useState(3)
  const [runId, setRunId] = useState('')
  const [error, setError] = useState('')
  const [viewing, setViewing] = useState<RunSummary | null>(null)
  const [histOpen, setHistOpen] = useState(true)
  const [est, setEst] = useState<{ need_mb: number; free_mb: number; ok: boolean; device?: string } | null>(null)
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
  useEffect(() => {
    const m = prepared.find((p) => p.id === datasetId)
    const langs = m?.languages ?? []
    if (langs.length === 1) setLanguage(langs[0].toLowerCase())
    else if (langs.length > 1) setLanguage('')
  }, [datasetId, prepared])

  useEffect(() => {
    api.estimate({ kind: 'train', model, method }).then(setEst).catch(() => setEst(null))
  }, [model, method, busy])

  const spec = models.find((m) => m.id === model)
  const ds = prepared.find((p) => p.id === datasetId)
  const canTrain = !!spec?.can_train
  const stepsPerEpoch = ds ? Math.max(1, Math.floor(ds.train / Math.max(1, batch * accum))) : 0
  const ready = !!datasetId && canTrain

  async function run() {
    if (!ready) return
    setError('')
    setBusy(true)
    setLosses([])
    setEvals([])
    setLastEval(null)
    setLive(null)
    setRunId('')
    setViewing(null)
    setStatus('starting')
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.trainStream(
        { dataset_id: datasetId, model, method, max_steps: maxSteps, batch_size: batch, grad_accum: accum,
          learning_rate: lr, warmup_steps: warmup, name, fp16: hw?.device === 'cuda',
          language: language || 'burmese',
          eval_steps: evalEvery, eval_clips: evalClips,
          early_stopping: earlyStop, patience },
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
            <label title="Which language Whisper is told to transcribe. Set automatically from the dataset.">language</label>
            <select value={language} onChange={(e) => setLanguage(e.target.value)} style={{ width: 150 }}>
              <option value="burmese">Burmese</option>
              <option value="thai">Thai</option>
              <option value="english">English</option>
              <option value="chinese">Chinese</option>
              <option value="">(multilingual)</option>
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
                  {m.burmese === true ? (
                    <span className="tag ok">knows Burmese</span>
                  ) : m.burmese === false ? (
                    <span className="tag warn">no Burmese ({m.languages} langs)</span>
                  ) : null}
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
            <label title="Score the model on held-out clips every N steps. 0 = automatic, about six checks across the run. This does not affect what the model trains on.">
              validate every
            </label>
            <input type="number" min={0} value={evalEvery} onChange={(e) => setEvalEvery(+e.target.value)} style={{ width: 80 }} />
            <span className="small">steps, on</span>
            <input type="number" min={1} max={64} value={evalClips} onChange={(e) => setEvalClips(+e.target.value)} style={{ width: 70 }} title="How many held-out clips each validation check transcribes. More gives a steadier number but each check takes longer. Training always uses the whole training split." />
            <span className="small">held-out clips {ds ? `(of ${ds.test} kept aside)` : ''}</span>
            <label title="Stop when the held-out error rate stops improving, and export the best checkpoint rather than the last.">
              <input type="checkbox" checked={earlyStop} onChange={(e) => setEarlyStop(e.target.checked)} style={{ width: 'auto', marginRight: 6 }} />
              early stop
            </label>
            {earlyStop && (
              <>
                <label title="How many checks without improvement to tolerate before stopping.">patience</label>
                <input type="number" min={1} max={20} value={patience} onChange={(e) => setPatience(+e.target.value)} style={{ width: 70 }} />
              </>
            )}
          </div>
          <div className="row">
            <label>run name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="optional" style={{ width: 240 }} />
            <span className="small">
              effective batch {batch * accum}
              {ds ? ` · trains on all ${ds.train.toLocaleString()} clips · ${stepsPerEpoch.toLocaleString()} steps = 1 epoch` : ''}
            </span>
          </div>
          <div className="row">
            <button className="primary" disabled={busy || !ready || (!!est && !est.ok)} onClick={run} title={!datasetId ? 'Pick a prepared dataset' : !canTrain ? `${spec?.label} cannot be fine-tuned here` : est && !est.ok ? 'Not enough memory for this combination' : ''}>
              {busy ? 'Training…' : 'Start fine-tuning'}
            </button>
            {busy && (
              <>
                <button className="ghost" onClick={() => runId && api.runStop(runId)}>Stop after this step</button>
                <button className="ghost" onClick={() => abortRef.current?.abort()}>Abort</button>
              </>
            )}
            {!canTrain && spec && <span className="small" style={{ color: 'var(--bad)' }}>{spec.label} is inference only here. {spec.note}</span>}
            {canTrain && est && (
              <span className="small" style={{ color: est.ok ? undefined : 'var(--bad)' }} title="Estimated working set: weights, plus gradients and optimiser state for a full fine-tune">
                needs ~<b>{(est.need_mb / 1024).toFixed(1)} GB</b> of {est.device === 'cuda' ? 'VRAM' : 'RAM'} · {(est.free_mb / 1024).toFixed(1)} GB free
                {est.ok ? ' ✓' : ' — the backend will refuse this rather than run out of memory'}
              </span>
            )}
          </div>
          {canTrain && spec?.burmese === false && (
            <div className="small" style={{ color: '#ffb454' }}>
              ⚠ {spec.label} was trained on {spec.languages} languages and Burmese is not one of them, so it has no Burmese
              tokens to build on. A Whisper model will reach a usable error rate on far less data.
            </div>
          )}
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
                  {lastEval?.best_cer != null && (
                    <span title={`best held-out score so far, at step ${lastEval.best_step}`}>
                      best <b style={{ color: 'var(--ok)' }}>{(lastEval.best_cer * 100).toFixed(1)}%</b>
                      {lastEval.stale > 0 ? ` · ${lastEval.stale}/${lastEval.patience} checks without improvement` : ''}
                    </span>
                  )}
                  {lastEval && (
                    <span title={`scored on ${lastEval.n} held-out clips at step ${lastEval.step}`}>
                      held-out CER <b style={{ color: evals.length > 1 && lastEval.cer <= evals[0].cer ? 'var(--ok)' : undefined }}>
                        {(lastEval.cer * 100).toFixed(1)}%
                      </b> · WER <b>{(lastEval.wer * 100).toFixed(1)}%</b>
                    </span>
                  )}
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
            {(evals.length > 0 || (viewing?.evals?.length ?? 0) > 0) && (
              <div style={{ marginTop: 14 }}>
                <LossChart
                  points={(viewing?.evals ?? evals).map((e) => ({ step: e.step, loss: e.cer * 100 }))}
                  title="Held-out character error rate (%) — lower is better"
                />
              </div>
            )}
            {lastEval?.sample && (
              <details style={{ marginTop: 8 }}>
                <summary>Latest held-out example (step {lastEval.step})</summary>
                <div className="clip">
                  <div className="small">reference</div>
                  <div className="my">{lastEval.sample.reference}</div>
                  <div className="small" style={{ marginTop: 6 }}>model output</div>
                  <div className="my">{lastEval.sample.hypothesis || <span className="small">(empty)</span>}</div>
                </div>
              </details>
            )}
            {shown && !viewing && runId && <div className="small">Run id {runId}. It appears in Evaluate as a fine-tuned target once finished.</div>}
          </section>
        )}
      </div>
    </div>
  )
}
