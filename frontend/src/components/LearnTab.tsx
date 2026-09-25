import { marked } from 'marked'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import LangPicker from './LangPicker'

interface Doc {
  file: string
  title: string
  summary: string
  chars: number
}
interface Msg {
  role: 'user' | 'assistant'
  content: string
  sources?: { n: number; title: string; file: string }[]
}

const STARTERS = [
  'How much audio do I need to fine-tune a model?',
  'LoRA or full fine-tuning — which should I use?',
  'Why is my error rate above 100%?',
  'What learning rate should I start with?',
  'How do I know when to stop training?',
  'My language is not in the model. What now?',
]

function md(text: string) {
  return { __html: marked.parse(text, { async: false }) as string }
}

export default function LearnTab({ aiEnabled, textModel }: { aiEnabled: boolean; textModel: string }) {
  const [docs, setDocs] = useState<Doc[]>([])
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState<{ stage: string; message: string } | null>(null)
  const [error, setError] = useState('')
  const [viewing, setViewing] = useState<{ file: string; content: string } | null>(null)
  const [lang, setLang] = useState<string>(() => {
    try {
      return localStorage.getItem('asr-learn-lang') ?? 'Auto'
    } catch {
      return 'Auto'
    }
  })
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.learnDocs().then(setDocs).catch(() => setDocs([]))
  }, [])
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs, busy])
  useEffect(() => {
    if (!viewing) return
    const h = (e: KeyboardEvent) => e.key === 'Escape' && setViewing(null)
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [viewing])

  async function ask(text: string) {
    const q = text.trim()
    if (!q || busy) return
    setError('')
    const next: Msg[] = [...msgs, { role: 'user', content: q }]
    setMsgs(next)
    setInput('')
    setBusy(true)
    setStage({ stage: 'selecting', message: 'starting…' })
    try {
      const r = await api.learnAsk(
        q,
        msgs.map((m) => ({ role: m.role, content: m.content })),
        (stg, message) => setStage({ stage: stg, message }),
        lang === 'Auto' ? undefined : lang,
      )
      setMsgs([...next, { role: 'assistant', content: r.answer, sources: r.sources }])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
      setStage(null)
    }
  }

  return (
    <div className="chatwrap" style={{ gridTemplateColumns: '280px 1fr' }}>
      <section className="panel">
        <h2>Handbook ({docs.length})</h2>
        <div className="small" style={{ marginBottom: 8 }}>
          The assistant answers only from these pages. Click one to read it.
        </div>
        <div className="sessions" style={{ maxHeight: '70vh' }}>
          {docs.map((d) => (
            <div
              key={d.file}
              className={`session ${viewing?.file === d.file ? 'on' : ''}`}
              onClick={() => api.learnDoc(d.file).then(setViewing).catch((e) => setError((e as Error).message))}
              title={d.summary}
            >
              <span>{d.title}</span>
            </div>
          ))}
        </div>
      </section>

      <div>
        <section className="panel">
          <h2 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span>Learn about fine-tuning speech models</span>
            <span style={{ textTransform: 'none', letterSpacing: 0, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
              answer in
              <LangPicker
                value={lang}
                onChange={(v) => {
                  setLang(v)
                  try {
                    localStorage.setItem('asr-learn-lang', v)
                  } catch {
                    /* ignore */
                  }
                }}
              />
            </span>
          </h2>
          {!aiEnabled && (
            <div className="error">
              Add an OpenRouter key in Settings to ask questions. You can still read the handbook on the left.
            </div>
          )}
          <div className="chat">
            {msgs.length === 0 && (
              <div className="msg assistant">
                <p>
                  Ask anything about fine-tuning speech recognition: datasets, LoRA against full fine-tuning,
                  hyperparameters, error metrics, hardware, or working with a low-resource language. Ask in any
                  language and I will answer in yours, using only the handbook, with citations.
                </p>
                <div className="suggest">
                  {STARTERS.map((s) => (
                    <button key={s} className="chip" onClick={() => ask(s)} disabled={!aiEnabled}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div dangerouslySetInnerHTML={md(m.content)} />
                {m.sources && m.sources.length > 0 && (
                  <div className="small" style={{ marginTop: 6 }}>
                    Sources:{' '}
                    {m.sources.map((s) => (
                      <button
                        key={s.n}
                        className="chip"
                        style={{ padding: '1px 8px', fontSize: 11, marginRight: 4 }}
                        onClick={() => api.learnDoc(s.file).then(setViewing)}
                      >
                        [{s.n}] {s.title}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {busy && (
              <div className="msg assistant">
                <div className="stage">
                  {['selecting', 'reading', 'answering'].map((st, idx) => (
                    <span
                      key={st}
                      className={`step ${stage?.stage === st ? 'on' : ''} ${
                        idx < ['selecting', 'reading', 'answering'].indexOf(stage?.stage ?? '') ? 'done' : ''
                      }`}
                    >
                      {st}
                    </span>
                  ))}
                </div>
                <div className="small">{stage?.message}</div>
              </div>
            )}
            <div ref={bottom} />
          </div>
          <div className="composer">
            <textarea
              placeholder={`Ask about fine-tuning, datasets, LoRA, metrics…  (${textModel || 'text model'})`}
              value={input}
              disabled={!aiEnabled || busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  ask(input)
                }
              }}
            />
            <button className="primary" disabled={!aiEnabled || busy || !input.trim()} onClick={() => ask(input)}>
              Ask
            </button>
            {msgs.length > 0 && (
              <button className="ghost" onClick={() => setMsgs([])} disabled={busy}>
                Clear
              </button>
            )}
          </div>
          {error && <div className="error">{error}</div>}
        </section>

        {viewing && (
          <div className="modal-bg" onClick={() => setViewing(null)}>
            <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
              <header>
                <b>{docs.find((d) => d.file === viewing.file)?.title ?? viewing.file}</b>
                <button className="chip" onClick={() => setViewing(null)}>
                  ✕ close
                </button>
              </header>
              <div className="body" dangerouslySetInnerHTML={md(viewing.content)} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
