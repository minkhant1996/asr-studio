import type { DatasetSourceInfo, DatasetSpec, EvalRecord, Manifest, ModelSpec, PreviewResult, RunSummary, Source, SystemInfo, Target } from './types'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    headers: init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!r.ok) {
    let msg: unknown = r.statusText
    try {
      msg = (await r.json()).detail ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return r.json()
}

/** Reads an NDJSON stream, calling `onEvent` per line. Resolves when the stream ends. */
async function ndjson(path: string, body: unknown, onEvent: (e: any) => void, signal?: AbortSignal) {
  const r = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText)
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (line) onEvent(JSON.parse(line))
    }
  }
}

export const api = {
  health: () => req<{ ok: boolean; device: string; hf_token_set: boolean }>('/health'),
  system: () => req<SystemInfo>('/system'),

  datasets: () => req<DatasetSpec[]>('/datasets'),
  datasetSourceInfo: (source: { dataset_id?: string; path?: string }) =>
    req<DatasetSourceInfo>(`/datasets/source-info?${new URLSearchParams(source.dataset_id ? { dataset_id: source.dataset_id } : { path: source.path ?? '' }).toString()}`),
  preview: (body: Record<string, unknown>) => req<PreviewResult>('/datasets/preview', { method: 'POST', body: JSON.stringify(body) }),
  prepareStream: (body: Record<string, unknown>, onEvent: (e: any) => void, signal?: AbortSignal) =>
    ndjson('/prepare/stream', body, onEvent, signal),
  prepared: () => req<Manifest[]>('/prepared'),
  preparedGet: (id: string) => req<Manifest>(`/prepared/${id}`),
  preparedDelete: (id: string) => req<{ ok: boolean }>(`/prepared/${id}`, { method: 'DELETE' }),
  preparedRows: (id: string, split: string, offset = 0, limit = 25) =>
    req<{ split: string; total: number; rows: { index: number; text: string; duration: number; source: string }[] }>(
      `/prepared/${id}/rows?split=${split}&offset=${offset}&limit=${limit}`,
    ),
  clipUrl: (id: string, split: string, index: number) => `/api/prepared/${id}/clip/${split}/${index}`,

  models: () => req<{ models: ModelSpec[]; device: string; gpu_gb: number; free_ram_gb: number }>('/models'),
  estimate: (body: Record<string, unknown>) =>
    req<{ kind: string; need_mb: number; free_mb: number; disk_mb?: number; ok: boolean; device?: string; note?: string }>('/estimate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  trainStream: (body: Record<string, unknown>, onEvent: (e: any) => void, signal?: AbortSignal) =>
    ndjson('/train/stream', body, onEvent, signal),
  runs: () => req<RunSummary[]>('/runs'),
  runGet: (id: string) => req<RunSummary>(`/runs/${id}`),
  runStop: (id: string) => req<{ ok: boolean }>(`/runs/${id}/stop`, { method: 'POST' }),
  runDelete: (id: string) => req<{ ok: boolean }>(`/runs/${id}`, { method: 'DELETE' }),

  evaluateStream: (body: Record<string, unknown>, onEvent: (e: any) => void, signal?: AbortSignal) =>
    ndjson('/evaluate/stream', body, onEvent, signal),
  evals: () => req<EvalRecord[]>('/evals'),
  evalGet: (id: string) => req<EvalRecord>(`/evals/${id}`),
  evalDelete: (id: string) => req<{ ok: boolean }>(`/evals/${id}`, { method: 'DELETE' }),

  transcribe: async (file: File, target: Target, language = 'burmese') => {
    const fd = new FormData()
    fd.append('file', file)
    const url = `/api/transcribe?target=${encodeURIComponent(JSON.stringify(target))}&language=${encodeURIComponent(language)}`
    const r = await fetch(url, { method: 'POST', body: fd })
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText)
    return (await r.json()) as {
      text: string
      chunks: string[]
      duration: number
      ms: number
      rtf: number | null
      model: string
      device: string
      filename: string
    }
  },

  settings: () => req<{
    hf_token_set: boolean
    hf_token_masked: string | null
    openrouter_key_set: boolean
    openrouter_key_masked: string | null
    openrouter_model: string
    prefs: Record<string, unknown>
    cache_dir: string
    runs_dir: string
  }>('/settings'),
  setOpenRouter: (api_key: string, model?: string) =>
    req<{ ok: boolean; masked: string; label?: string; model: string }>('/settings/openrouter', {
      method: 'PUT',
      body: JSON.stringify({ api_key, model }),
    }),
  clearOpenRouter: () => req<{ ok: boolean }>('/settings/openrouter', { method: 'DELETE' }),
  setOrModel: (openrouter_model: string) => req<{ ok: boolean; model: string }>('/settings/model', { method: 'PUT', body: JSON.stringify({ openrouter_model }) }),
  orModels: () => req<{ id: string; name: string; context: number | null; prompt_price: number; completion_price: number }[]>('/openrouter/models'),
  learnDocs: () => req<{ file: string; title: string; summary: string; chars: number }[]>('/learn/docs'),
  learnDoc: (file: string) => req<{ file: string; content: string }>(`/learn/docs/${file}`),
  learnAsk: async (
    question: string,
    history: { role: string; content: string }[],
    onStatus: (stage: string, message: string) => void,
    language?: string,
  ) => {
    const r = await fetch('/api/learn/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history, language }),
    })
    if (!r.ok || !r.body) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText)
    const reader = r.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    let done: { answer: string; sources: { n: number; title: string; file: string }[] } | null = null
    for (;;) {
      const { value, done: end } = await reader.read()
      if (end) break
      buf += dec.decode(value, { stream: true })
      let nl
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (!line) continue
        const ev = JSON.parse(line)
        if (ev.type === 'status') onStatus(ev.stage, ev.message)
        else if (ev.type === 'error') throw new Error(ev.message)
        else if (ev.type === 'done') done = ev
      }
    }
    if (!done) throw new Error('stream ended without an answer')
    return done
  },
  setHfToken: (token: string) => req<{ ok: boolean; user: string; masked: string }>('/settings/hf', { method: 'PUT', body: JSON.stringify({ token }) }),
  clearHfToken: () => req<{ ok: boolean }>('/settings/hf', { method: 'DELETE' }),
}

export type { Source }
