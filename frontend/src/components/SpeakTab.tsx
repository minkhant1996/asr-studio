import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { SpeakResult, TtsStatus } from '../types'

const VOICES = [
  'A young woman, gentle and warm voice',
  'A middle-aged man, calm news-reader voice',
  'An elderly man, slow and kind',
  'Cheerful, slightly faster',
]

const MODES: Record<SpeakResult['mode'], string> = {
  default: 'a voice the model chose',
  design: 'a voice designed from the description',
  clone: 'the voice of the reference clip',
  'clone+transcript': 'the reference clip, continued from its transcript',
}

export default function SpeakTab() {
  const [status, setStatus] = useState<TtsStatus | null>(null)
  const [model, setModel] = useState('')
  const [text, setText] = useState('မင်္ဂလာပါ။ ဒီနေ့ ရာသီဥတု အရမ်းကောင်းပါတယ်။')
  const [voice, setVoice] = useState('')
  const [reference, setReference] = useState<File | null>(null)
  const [refUrl, setRefUrl] = useState('')
  const [promptText, setPromptText] = useState('')
  const [cfg, setCfg] = useState(2)
  const [steps, setSteps] = useState(10)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState<SpeakResult | null>(null)
  const [url, setUrl] = useState('')
  const refInput = useRef<HTMLInputElement>(null)

  const loadStatus = (id?: string) => api.tts(id || model || undefined).then(setStatus).catch(() => setStatus(null))
  useEffect(() => {
    loadStatus()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  useEffect(() => () => { if (refUrl) URL.revokeObjectURL(refUrl) }, [refUrl])
  useEffect(() => {
    if (!busy) return
    const t0 = Date.now()
    const t = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000)
    return () => clearInterval(t)
  }, [busy])

  function pickReference(f: File | null | undefined) {
    setReference(f ?? null)
    setRefUrl(f ? URL.createObjectURL(f) : '')
    if (!f) setPromptText('')
    if (refInput.current) refInput.current.value = ''
  }

  async function run() {
    setBusy(true)
    setElapsed(0)
    setError('')
    try {
      const r = await api.speak({ text, model: status?.id, voice, reference, promptText, cfg, steps })
      const bytes = Uint8Array.from(atob(r.audio_b64), (c) => c.charCodeAt(0))
      setUrl(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })))
      setResult(r)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
      loadStatus()
    }
  }

  const maxText = status?.max_text ?? 1000
  const transcriptGiven = !!reference && !!promptText.trim()
  const unsupported = status?.status === 'unsupported'
  const blocked = status ? unsupported || !status.installed || !status.fits : false

  return (
    <div className="grid">
      <section className="panel">
        <h2>Speak text aloud</h2>
        <div className="chips" style={{ marginBottom: 8 }}>
          {(status?.models ?? []).map((m) => (
            <button
              key={m.id}
              className={`chip ${status?.id === m.id ? 'on' : ''}`}
              disabled={busy}
              title={m.note}
              onClick={() => setModel(m.id)}
            >
              {m.label}
              {m.status === 'unsupported' ? ' · unavailable' : m.burmese ? ' · Burmese' : ''}
            </button>
          ))}
        </div>
        {status && (
          <div className="small" style={{ marginBottom: 10 }}>
            <a href={status.url} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>{status.label}</a> · {status.params} · {status.languages} languages
            {status.burmese ? ' including Burmese and Thai' : ' (no Burmese)'} · {status.sample_rate / 1000} kHz · {status.license}
            <br />
            {status.note}
            <br />
            Text-to-speech, the reverse of the other tabs: these cannot transcribe, so they are not in the base-model lists.
          </div>
        )}
        {unsupported && (
          <div className="error">
            {status?.label} cannot run in this backend, so it is listed but not selectable for synthesis.
          </div>
        )}
        {status && !unsupported && !status.installed && (
          <div className="error">
            {status.label} needs a package that is not installed in the backend environment. Run this in <code>backend/</code>, then restart the backend:
            <pre>{status.install_hint}</pre>
          </div>
        )}
        {status?.installed && !unsupported && !status.fits && (
          <div className="error">
            {status.label} needs about {(status.need_mb / 1024).toFixed(0)} GB of {status.device === 'cuda' ? 'VRAM' : 'RAM'} and only{' '}
            {((status.free_mb ?? 0) / 1024).toFixed(1)} GB is free.
          </div>
        )}
        <textarea
          className="my"
          value={text}
          maxLength={maxText}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste the text to speak, in any supported language"
          style={{ minHeight: 120, fontSize: 15, lineHeight: 1.6, resize: 'vertical' }}
        />
        <div className="small">{text.length} / {maxText} characters · no language setting needed</div>

        {status?.design !== false && (
          <>
            <div className="row"><label>voice description (optional)</label></div>
            <input
              value={voice}
              disabled={transcriptGiven}
              onChange={(e) => setVoice(e.target.value)}
              placeholder="e.g. A young woman, gentle and warm voice"
              style={{ marginTop: 6 }}
            />
            <div className="chips" style={{ marginTop: 6 }}>
              {VOICES.map((v) => (
                <button key={v} className={`chip ${voice === v ? 'on' : ''}`} disabled={transcriptGiven} onClick={() => setVoice(voice === v ? '' : v)}>
                  {v}
                </button>
              ))}
            </div>
            <div className="small" style={{ marginTop: 6 }}>
              {reference
                ? 'With a reference clip, the description steers style (pace, emotion) while the clip sets the voice.'
                : 'Without a description the model picks a voice, and a different one each time. Write descriptions in English or Chinese.'}
            </div>
          </>
        )}
        {status?.design === false && status.clone && (
          <div className="small" style={{ marginTop: 6 }}>
            {status.label} has no voice-description mode. Give it a reference clip to set the voice, or leave the clip
            out and it picks one.
          </div>
        )}

        {status?.clone === false ? (
          <div className="small" style={{ marginTop: 6 }}>
            {status.label} cannot clone a voice: its speakers ship as fixed embeddings.
          </div>
        ) : (
          <>
            <div className="row"><label>reference clip to clone (optional)</label></div>
            <div className="row" style={{ marginTop: 6 }}>
              <button className="ghost" onClick={() => refInput.current?.click()}>{reference ? 'Replace clip' : 'Choose a clip'}</button>
              {reference && <span className="small">{reference.name}</span>}
              {reference && <button className="chip" onClick={() => pickReference(null)}>Remove</button>}
            </div>
            <input ref={refInput} type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.opus,.webm,.aac" style={{ display: 'none' }} onChange={(e) => pickReference(e.target.files?.[0])} />
            {refUrl && <audio controls src={refUrl} style={{ marginTop: 8 }} />}
            {reference && (
              <>
                {status?.design !== false && (
                  <textarea
                    className="my"
                    value={promptText}
                    onChange={(e) => setPromptText(e.target.value)}
                    placeholder="Exact transcript of the clip (optional): closer cloning, but the voice description is then ignored"
                    style={{ marginTop: 8, minHeight: 60, fontSize: 14, resize: 'vertical' }}
                  />
                )}
                <div className="small">
                  A few seconds of clean speech is enough; only the first {status?.max_reference_seconds ?? 30} s are used. Clone only voices you have permission to use, and label the result as AI-generated.
                </div>
              </>
            )}
          </>
        )}

        {status?.advanced !== false && (
          <details>
            <summary>Advanced</summary>
            <div className="row">
              <label title="How closely the audio follows the text and voice prompt. Higher is more literal, lower more natural.">guidance {cfg.toFixed(1)}</label>
              <input type="range" min={1} max={3} step={0.1} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} style={{ width: 160, padding: 0 }} />
              <label title="Diffusion steps per audio frame. More is slower and a little cleaner.">steps {steps}</label>
              <input type="range" min={4} max={30} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ width: 160, padding: 0 }} />
            </div>
          </details>
        )}

        <div className="row">
          <button className="primary" disabled={busy || !text.trim() || blocked} onClick={run}>{busy ? `Speaking… ${elapsed}s` : 'Speak'}</button>
          {status?.installed && status.device !== 'cuda' && (
            <span className="small">Running on {status.device.toUpperCase()}: expect roughly 5 s of work per second of speech.</span>
          )}
        </div>
        {busy && !status?.loaded && <div className="small">First use downloads about 4.7 GB of weights and loads them, which can take several minutes.</div>}
        {error && <div className="error">{error}</div>}
      </section>
      <section className="panel">
        <h2>Audio</h2>
        {!result && <div className="small">The generated speech appears here, playable and downloadable as a WAV file.</div>}
        {result && (
          <>
            <audio controls autoPlay src={url} />
            <div className="metrics" style={{ flexWrap: 'wrap' }}>
              <div className="metric"><div className="k">audio</div><div className="v" style={{ fontSize: 18 }}>{result.duration}s</div></div>
              <div className="metric"><div className="k">took</div><div className="v" style={{ fontSize: 18 }}>{(result.ms / 1000).toFixed(2)}s</div></div>
              <div className="metric"><div className="k">real-time factor</div><div className="v" style={{ fontSize: 18 }}>{result.rtf ?? '—'}</div></div>
              <div className="metric"><div className="k">device</div><div className="v" style={{ fontSize: 16 }}>{result.device}</div></div>
            </div>
            <div className="small">{result.model} · {result.sampling_rate / 1000} kHz · {MODES[result.mode]}</div>
            <div className="row">
              <a className="chip" href={url} download="speech.wav">Download WAV</a>
            </div>
            <div className="small" style={{ marginTop: 10 }}>
              Results vary between runs, most of all with a voice description alone. If one sounds off, generate again.
            </div>
          </>
        )}
      </section>
    </div>
  )
}
