import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { DatasetSourceInfo, DatasetSpec, Manifest, PreviewResult, Source } from '../types'
import { useConfirm } from './ConfirmDialog'

const fmt = (n: number) => n.toLocaleString()
const sourceKey = (source: Source) => source.dataset_id ?? source.path ?? ''

function sourceMax(source: Source, info?: DatasetSourceInfo): number | null {
  if (!info) return null
  const counts = new Map(info.splits.map((split) => [split.name, split.max_clips]))
  if (source.split === 'both') {
    const train = counts.get('train')
    const test = counts.get('test')
    return train != null && test != null ? train + test : null
  }
  return counts.get(source.split) ?? null
}

const quickPercents = [25, 60, 75, 100] as const

export default function DataTab({ onPrepared, prepared, reload }: { onPrepared: (m: Manifest) => void; prepared: Manifest[]; reload: () => void }) {
  const [specs, setSpecs] = useState<DatasetSpec[]>([])
  const [mix, setMix] = useState<Source[]>([])
  const [sourceInfos, setSourceInfos] = useState<Record<string, DatasetSourceInfo>>({})
  const [infoLoading, setInfoLoading] = useState<Record<string, boolean>>({})
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [previewing, setPreviewing] = useState('')
  const [hfRef, setHfRef] = useState('')
  const [hfDir, setHfDir] = useState('train')
  const [name, setName] = useState('')
  const [minS, setMinS] = useState(0.4)
  const [maxS, setMaxS] = useState(20)
  const [testFrac, setTestFrac] = useState(0.1)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [prog, setProg] = useState<{ i: number; n: number; seconds: number; elapsed: number; eta: number; skipped: number } | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Manifest | null>(null)
  const [rows, setRows] = useState<{ index: number; text: string; duration: number; source: string }[]>([])
  const [rowSplit, setRowSplit] = useState('train')
  const abortRef = useRef<AbortController | null>(null)
  const { confirm, dialog } = useConfirm()

  useEffect(() => {
    api.datasets().then(setSpecs).catch(() => setSpecs([]))
  }, [])
  useEffect(() => {
    if (!selected) return
    api.preparedRows(selected.id, rowSplit, 0, 25).then((r) => setRows(r.rows)).catch(() => setRows([]))
  }, [selected, rowSplit])

  const inMix = (id: string) => mix.some((m) => m.dataset_id === id)

  async function loadSourceInfo(source: Source) {
    const key = sourceKey(source)
    if (sourceInfos[key] || infoLoading[key]) return
    setInfoLoading((current) => ({ ...current, [key]: true }))
    try {
      const info = await api.datasetSourceInfo(source.dataset_id ? { dataset_id: source.dataset_id } : { path: source.path })
      setSourceInfos((current) => ({ ...current, [key]: info }))
      setMix((current) => current.map((item) => {
        if (sourceKey(item) !== key) return item
        const options = info.splits.map((entry) => entry.name)
        const split = options.includes(item.split) ? item.split : options[0] ?? item.split
        const next = { ...item, split }
        const max = sourceMax(next, info)
        return { ...next, take: max == null ? item.take : Math.min(item.take, max) }
      }))
    } catch {
      // Preparation still works when Hugging Face does not publish split counts.
    } finally {
      setInfoLoading((current) => ({ ...current, [key]: false }))
    }
  }

  function toggle(spec: DatasetSpec) {
    if (!spec.transcripts) return
    if (!inMix(spec.id)) loadSourceInfo({ dataset_id: spec.id, split: spec.splits[0], take: 100 })
    setMix((current) => current.some((item) => item.dataset_id === spec.id)
      ? current.filter((item) => item.dataset_id !== spec.id)
      : [...current, { dataset_id: spec.id, split: spec.splits[0], take: 100, label: spec.name }])
  }

  function addCustom() {
    const source: Source = { path: hfRef.trim(), data_dir: hfDir || null, split: 'train', take: 100, label: hfRef.trim() }
    setMix((current) => [...current, source])
    loadSourceInfo(source)
  }

  async function doPreview(spec?: DatasetSpec) {
    setError('')
    setPreview(null)
    setPreviewing(spec ? spec.id : 'custom')
    try {
      setPreview(spec ? await api.preview({ dataset_id: spec.id, limit: 3 }) : await api.preview({ path: hfRef, data_dir: hfDir || null, split: 'train', limit: 3 }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPreviewing('')
    }
  }

  async function run() {
    if (!mix.length) {
      setError('Add at least one dataset to the mix.')
      return
    }
    if (mix.some((source) => source.take < 1)) {
      setError('Remove any source with zero available clips or choose another split.')
      return
    }
    setError('')
    setBusy(true)
    setProg(null)
    setStatus('starting')
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.prepareStream(
        { sources: mix.map(({ label, ...s }) => s), name, min_seconds: minS, max_seconds: maxS, test_fraction: testFrac },
        (ev) => {
          if (ev.type === 'status') setStatus(ev.message)
          else if (ev.type === 'progress') {
            setProg({ i: ev.i, n: ev.n, seconds: ev.seconds, elapsed: ev.elapsed, eta: ev.eta, skipped: ev.skipped })
            setStatus(`streaming ${ev.source}`)
          } else if (ev.type === 'done') {
            setStatus('done')
            onPrepared(ev.manifest)
            setSelected(ev.manifest)
            reload()
          } else if (ev.type === 'error') setError(ev.message)
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

  const totalTake = mix.reduce((s, m) => s + (m.take || 0), 0)
  const diskGb = (totalTake * 6 * 32 * 1024) / 2 ** 30   // PCM16 @16 kHz, ~6 s average clip

  return (
    <div>
      {dialog}
      <section className="panel">
        <h2>1 · Pick speech datasets</h2>
        {[...new Set(specs.map((x) => x.language))].map((lang) => (
        <div key={lang} style={{ marginBottom: 12 }}>
        <div className="small" style={{ margin: '4px 0 6px', fontWeight: 600 }}>
          {lang} · {specs.filter((x) => x.language === lang).length} datasets
        </div>
        <div className="cards">
          {specs.filter((x) => x.language === lang).map((s) => (
            <div key={s.id} className={`card ${inMix(s.id) ? 'on' : ''}`} onClick={() => toggle(s)} title={s.transcripts ? 'Click to add to the mix' : 'Audio only: cannot be used for supervised training'}>
              <b>{s.name}</b>
              <div className="small">{s.description}</div>
              <div className="meta">
                <span className="tag">{s.hours} h</span>
                <span className="tag">{s.clips} clips</span>
                <span className="tag">{s.license}</span>
                {s.transcripts ? <span className="tag ok">transcripts</span> : <span className="tag warn">audio only</span>}
                {s.streaming_only && <span className="tag">streamed</span>}
                {s.slow_stream && <span className="tag warn">slow: one request per clip</span>}
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button className="chip" onClick={(e) => { e.stopPropagation(); doPreview(s) }} disabled={!!previewing}>
                  {previewing === s.id ? 'loading…' : '▶ preview'}
                </button>
                <a className="chip" href={s.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} style={{ textDecoration: 'none' }}>
                  card ↗
                </a>
              </div>
            </div>
          ))}
        </div>
        </div>
        ))}
        <div className="row">
          <label>or any Hugging Face dataset</label>
          <input value={hfRef} onChange={(e) => setHfRef(e.target.value)} placeholder="owner/name or a huggingface.co/datasets URL" style={{ flex: 1, minWidth: 220 }} />
          <input value={hfDir} onChange={(e) => setHfDir(e.target.value)} placeholder="data_dir (webdataset: train)" style={{ width: 170 }} />
          <button className="ghost" disabled={!hfRef.trim() || !!previewing} onClick={() => doPreview()}>
            {previewing === 'custom' ? 'loading…' : 'Preview'}
          </button>
          <button
            className="ghost"
            disabled={!hfRef.trim()}
            onClick={addCustom}
          >
            Add to mix
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {preview && (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>Preview · {preview.path}</h2>
          <div className="small" style={{ marginBottom: 8 }}>columns: {preview.columns.join(', ')}</div>
          {!preview.transcripts && <div className="error">This dataset has no transcripts, so it cannot be used for supervised fine-tuning.</div>}
          {preview.items.map((it, i) => (
            <div key={i} className="clip">
              <div className="my">{it.text || <span className="small">(no transcript)</span>}</div>
              <div className="small">{it.duration != null ? `${it.duration}s` : ''} {it.audio_error ? `· ${it.audio_error}` : ''}</div>
              {it.audio_b64 && <audio controls preload="none" src={`data:audio/wav;base64,${it.audio_b64}`} />}
            </div>
          ))}
        </section>
      )}

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>2 · Build a training set ({mix.length} source{mix.length === 1 ? '' : 's'}, {fmt(totalTake)} clips)</h2>
        {mix.length === 0 && <div className="small">Click a dataset above to add it. Mixing sources usually generalises better than one source alone.</div>}
        {mix.map((m, i) => {
          const spec = specs.find((item) => item.id === m.dataset_id)
          const key = sourceKey(m)
          const info = sourceInfos[key]
          const availableSplits = info?.splits.map((item) => item.name) ?? spec?.splits ?? [m.split]
          const splitOptions = availableSplits.includes('train') && availableSplits.includes('test')
            ? [...availableSplits, 'both'] : availableSplits
          const max = sourceMax(m, info)
          return (
            <div key={i} className="srcrow">
              <div className="src-label">
                <span className="small">{m.label ?? m.dataset_id ?? m.path}</span>
                <span className="small">
                  {max != null ? `Max ${fmt(max)} clips` : infoLoading[key] ? 'Checking source limit…' : `Max unavailable${spec ? ` · catalog ${spec.clips}` : ''}`}
                </span>
              </div>
              <label className="src-control">Source split
                <select value={m.split} onChange={(e) => {
                  const split = e.target.value
                  setMix((current) => current.map((item, j) => {
                    if (j !== i) return item
                    const next = { ...item, split }
                    const limit = sourceMax(next, info)
                    return { ...next, take: limit == null ? item.take : Math.min(item.take, limit) }
                  }))
                }}>
                  {splitOptions.map((split) => <option key={split} value={split}>{split === 'both' ? 'train + test' : split}</option>)}
                </select>
              </label>
              <label className="src-control">Clips to take
                <input
                  type="number"
                  min={max === 0 ? 0 : 1}
                  max={max ?? undefined}
                  value={m.take}
                  onChange={(e) => {
                    const requested = Math.max(1, Math.floor(Number(e.target.value) || 1))
                    setMix((current) => current.map((item, j) => j === i
                      ? { ...item, take: max == null ? requested : Math.min(requested, max) } : item))
                  }}
                />
              </label>
              <div className="src-percent" aria-label="Choose percentage of available clips">
                {quickPercents.map((pct) => {
                  const count = max == null ? null : max === 0 ? 0 : Math.max(1, Math.floor(max * pct / 100))
                  return <button key={pct} className={`chip ${count != null && m.take === count ? 'on' : ''}`}
                    disabled={count == null || count === 0} title={count == null ? 'Source limit unavailable' : `${fmt(count)} clips`}
                    onClick={() => setMix((current) => current.map((item, j) => j === i ? { ...item, take: count! } : item))}>
                    {pct}%
                  </button>
                })}
              </div>
              <button className="iconbtn" onClick={() => setMix((current) => current.filter((_, j) => j !== i))} title="remove">✕</button>
            </div>
          )
        })}
        {mix.length > 0 && <div className="small">Maximums count source rows before transcript and duration filters. Source split chooses input rows; the test split below holds out a share after mixing.</div>}
        {new Set(mix.map((m) => specs.find((s) => s.id === m.dataset_id)?.language).filter(Boolean)).size > 1 && (
          <div className="small" style={{ color: '#ffb454' }}>
            ⚠ This mix spans more than one language. That is fine for a multilingual model, but a single-language
            set usually reaches a lower error rate.
          </div>
        )}
        <div className="row">
          <label>name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="optional" style={{ width: 200 }} />
          <label title="Clips shorter than this are dropped">min s</label>
          <input type="number" step={0.1} min={0.1} value={minS} onChange={(e) => setMinS(+e.target.value)} style={{ width: 80 }} />
          <label title="Whisper truncates at 30 s">max s</label>
          <input type="number" step={1} min={1} max={30} value={maxS} onChange={(e) => setMaxS(+e.target.value)} style={{ width: 80 }} />
          <label title="Held out for evaluation">test split</label>
          <input type="number" step={0.05} min={0} max={0.5} value={testFrac} onChange={(e) => setTestFrac(+e.target.value)} style={{ width: 80 }} />
          <button className="primary" disabled={busy || !mix.length} onClick={run}>
            {busy ? 'Preparing…' : 'Prepare dataset'}
          </button>
          {busy && <button className="ghost" onClick={() => abortRef.current?.abort()}>Stop</button>}
          {totalTake > 0 && (
            <span className="small" title="Clips are written to disk in shards, so memory stays flat however many you take">
              ≈<b>{diskGb < 1 ? `${Math.round(diskGb * 1024)} MB` : `${diskGb.toFixed(1)} GB`}</b> on disk · memory stays flat
            </span>
          )}
        </div>
        {(busy || prog) && (
          <div style={{ marginTop: 10 }}>
            <div className="live">
              <span>status: <b>{status}</b></span>
              {prog && (
                <>
                  <span>clip <b>{prog.i}</b> / {prog.n}</span>
                  <span>audio <b>{prog.seconds}s</b></span>
                  <span>elapsed <b>{prog.elapsed}s</b></span>
                  <span>ETA <b>{prog.eta}s</b></span>
                  {prog.skipped > 0 && <span>skipped <b>{prog.skipped}</b></span>}
                </>
              )}
            </div>
            <div className="progress">
              <div style={{ width: prog ? `${(prog.i / Math.max(prog.n, 1)) * 100}%` : '0%' }} />
            </div>
            <div className="small">
              Clips are streamed, decoded, resampled to 16 kHz and written to disk in batches, so memory stays flat however many you take.
              {prog?.skipped ? ` ${prog.skipped} clip(s) skipped (empty transcript or outside the duration limits).` : ''}
            </div>
          </div>
        )}
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>3 · Prepared datasets ({prepared.length})</h2>
        {prepared.length === 0 && <div className="small">Prepared sets appear here and are reused by training and evaluation.</div>}
        <div className="cards">
          {prepared.map((m) => (
            <div key={m.id} className={`card ${selected?.id === m.id ? 'on' : ''}`} onClick={() => setSelected(m)}>
              <b>{m.name}</b>
              <div className="meta">
                <span className="tag">{fmt(m.clips)} clips</span>
                <span className="tag">{m.hours} h</span>
                <span className="tag">{m.train}/{m.test} train/test</span>
                <span className="tag">avg {m.avg_duration}s</span>
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <span className="small" style={{ flex: 1 }}>{new Date(m.created * 1000).toLocaleString()}</span>
                <button
                  className="iconbtn"
                  title="Delete this prepared dataset"
                  onClick={async (e) => {
                    e.stopPropagation()
                    if (await confirm({ title: 'Delete this prepared dataset?', message: <span>“{m.name}” and its {fmt(m.clips)} cached clips will be removed.</span> })) {
                      await api.preparedDelete(m.id)
                      if (selected?.id === m.id) setSelected(null)
                      reload()
                    }
                  }}
                >
                  ✕
                </button>
              </div>
            </div>
          ))}
        </div>
        {selected && (
          <div style={{ marginTop: 12 }}>
            <div className="row" style={{ marginTop: 0 }}>
              <b className="small">{selected.name}</b>
              <span className="small">id {selected.id} · sources: {selected.sources.map((s) => `${s.label} (${s.clips})`).join(', ')}</span>
              <select value={rowSplit} onChange={(e) => setRowSplit(e.target.value)} style={{ width: 130 }}>
                <option value="train">train ({selected.train})</option>
                <option value="test">test ({selected.test})</option>
              </select>
            </div>
            <div className="scroll" style={{ maxHeight: 420 }}>
              <table>
                <thead>
                  <tr>
                    <th style={{ width: 50 }}>#</th>
                    <th>transcript</th>
                    <th style={{ width: 70 }}>seconds</th>
                    <th style={{ width: 210 }}>audio</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.index}>
                      <td className="small num">{r.index}</td>
                      <td className="my">{r.text}</td>
                      <td className="num small">{r.duration}</td>
                      <td>
                        <audio controls preload="none" src={api.clipUrl(selected.id, rowSplit, r.index)} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
