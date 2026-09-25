import { useEffect, useState } from 'react'
import { api } from '../api'
import SystemStats from './SystemStats'

export default function SettingsTab({ onChange }: { onChange: () => void }) {
  const [info, setInfo] = useState<Awaited<ReturnType<typeof api.settings>> | null>(null)
  const [token, setToken] = useState('')
  const [orKey, setOrKey] = useState('')
  const [orModel, setOrModel] = useState('')
  const [show, setShow] = useState(false)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = () => api.settings().then((i) => { setInfo(i); setOrModel(i.openrouter_model) }).catch(() => setInfo(null))
  useEffect(() => {
    refresh()
  }, [])

  async function save() {
    setBusy(true)
    setError('')
    setMsg('')
    try {
      const r = await api.setHfToken(token.trim())
      setToken('')
      setMsg(`Token ${r.masked} verified as ${r.user} and saved encrypted.`)
      await refresh()
      onChange()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <SystemStats />
      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Hugging Face token (optional)</h2>
        <div className="small" style={{ marginBottom: 10 }}>
          {info?.hf_token_set ? `Active token: ${info.hf_token_masked}` : 'All seven Burmese datasets and the listed models are public, so a token is optional. Add one for gated or private repositories, or to raise rate limits.'}
        </div>
        <form
          autoComplete="off"
          onSubmit={(e) => {
            e.preventDefault()
            if (token.trim()) save()
          }}
        >
          <input type={show ? 'text' : 'password'} autoComplete="off" spellCheck={false} placeholder="hf_…" value={token} onChange={(e) => setToken(e.target.value)} />
          <div className="row">
            <button className="primary" type="submit" disabled={busy || token.trim().length < 8}>{busy ? 'Verifying…' : 'Verify & save'}</button>
            <button className="ghost" type="button" onClick={() => setShow((s) => !s)}>{show ? 'Hide' : 'Show'}</button>
            {info?.hf_token_set && (
              <button className="ghost" type="button" onClick={async () => { await api.clearHfToken(); await refresh(); onChange() }}>Remove</button>
            )}
          </div>
        </form>
        {msg && <div className="small" style={{ color: 'var(--ok)', marginTop: 8 }}>{msg}</div>}
        {error && <div className="error">{error}</div>}
        <details>
          <summary>How it is protected</summary>
          <ul className="small">
            <li>The token is sent once to the local backend and never stored in the browser.</li>
            <li>It is verified with the Hub, then encrypted at rest (Fernet) in <code>backend/data/</code> with 0600 permissions.</li>
            <li>The API never returns it; only the last four characters are shown.</li>
          </ul>
        </details>
      </section>
      <section className="panel" style={{ marginTop: 16 }}>
        <h2>OpenRouter key (for the Learn tab)</h2>
        <div className="small" style={{ marginBottom: 10 }}>
          {info?.openrouter_key_set
            ? `Active key: ${info.openrouter_key_masked} · model ${info.openrouter_model}`
            : 'Only the Learn tab uses a text model. Everything else — datasets, training, evaluation, transcription — runs locally without it.'}
        </div>
        <form
          autoComplete="off"
          onSubmit={async (e) => {
            e.preventDefault()
            setBusy(true)
            setError('')
            setMsg('')
            try {
              const r = await api.setOpenRouter(orKey.trim(), orModel.trim() || undefined)
              setOrKey('')
              setMsg(`OpenRouter key ${r.masked} verified and saved encrypted.`)
              await refresh()
              onChange()
            } catch (e2) {
              setError((e2 as Error).message)
            } finally {
              setBusy(false)
            }
          }}
        >
          <input type={show ? 'text' : 'password'} autoComplete="off" placeholder="sk-or-v1-…" value={orKey} onChange={(e) => setOrKey(e.target.value)} />
          <div className="row">
            <label>model</label>
            <input value={orModel} onChange={(e) => setOrModel(e.target.value)} placeholder="anthropic/claude-sonnet-5" style={{ maxWidth: 280 }} />
            <button className="primary" type="submit" disabled={busy || orKey.trim().length < 20}>
              Verify &amp; save
            </button>
            {info?.openrouter_key_set && (
              <>
                <button
                  className="ghost"
                  type="button"
                  onClick={async () => {
                    await api.setOrModel(orModel.trim())
                    await refresh()
                    setMsg('Model updated.')
                  }}
                >
                  Update model only
                </button>
                <button className="ghost" type="button" onClick={async () => { await api.clearOpenRouter(); await refresh(); onChange() }}>
                  Remove
                </button>
              </>
            )}
          </div>
        </form>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Storage</h2>
        <table className="aligned">
          <tbody>
            <tr><td>prepared datasets</td><td className="small">{info?.cache_dir}</td></tr>
            <tr><td>training runs &amp; checkpoints</td><td className="small">{info?.runs_dir}</td></tr>
            <tr><td>model weights</td><td className="small">~/.cache/huggingface</td></tr>
          </tbody>
        </table>
        <div className="small">Everything stays on this machine and is git-ignored.</div>
      </section>
    </div>
  )
}
