import type { DatasetSpec, EvalRecord, Manifest, ModelSpec, PreviewResult, RunSummary, Source, SystemInfo, Target } from './types'

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

  settings: () => req<{ hf_token_set: boolean; hf_token_masked: string | null; prefs: Record<string, unknown>; cache_dir: string; runs_dir: string }>('/settings'),
  setHfToken: (token: string) => req<{ ok: boolean; user: string; masked: string }>('/settings/hf', { method: 'PUT', body: JSON.stringify({ token }) }),
  clearHfToken: () => req<{ ok: boolean }>('/settings/hf', { method: 'DELETE' }),
}

export type { Source }
