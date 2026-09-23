export interface DatasetSpec {
  id: string
  name: string
  path: string
  hours: string
  clips: string
  license: string
  description: string
  config: string | null
  data_dir: string | null
  splits: string[]
  text_field: string | null
  streaming_only: boolean
  transcripts: boolean
  url: string
}

export interface PreviewItem {
  text: string
  duration: number | null
  audio_b64?: string
  sampling_rate?: number
  audio_error?: string
}

export interface PreviewResult {
  path: string
  split: string
  columns: string[]
  items: PreviewItem[]
  text_field: string | null
  data_dir: string | null
  transcripts: boolean
}

export interface Source {
  dataset_id?: string
  path?: string
  split: string
  take: number
  data_dir?: string | null
  text_field?: string | null
  label?: string
}

export interface Manifest {
  id: string
  name: string
  created: number
  clips: number
  train: number
  test: number
  seconds: number
  hours: number
  sampling_rate: number
  sources: { label: string; path: string; split: string; clips: number; seconds: number }[]
  skipped: number
  chars: number
  avg_duration: number
  min_seconds: number
  max_seconds: number
  samples: { text: string; duration: number; source: string }[]
}

export interface ModelSpec {
  id: string
  path: string
  family: string
  params: string
  label: string
  finetune: string
  note: string
  url: string
  vram_full_gb?: number
  vram_lora_gb?: number
  ram_cpu_gb?: number
  can_train: boolean
  fits_lora: boolean
  fits_full: boolean
}

export interface RunSummary {
  id: string
  created: number
  status: string
  name: string
  model: string
  model_label: string
  dataset_id: string
  dataset_name: string
  clips: number
  device: string
  losses: { step: number; loss: number }[]
  config: Record<string, unknown>
  train_runtime?: number
  train_loss?: number
  steps?: number
  model_dir?: string
  message?: string
  stopped_early?: boolean
}

export interface EvalRow {
  reference: string
  hypothesis: string
  duration: number
  ms: number
  cer: number
}

export interface EvalTargetResult {
  label: string
  target: Record<string, unknown>
  device: string
  wer: number
  cer: number
  n: number
  avg_ms: number
  total_s: number
  rtf: number | null
  audio_s: number
  rows?: EvalRow[]
}

export interface EvalRecord {
  id: string
  created: number
  title: string
  dataset_id: string
  dataset_name: string
  split: string
  n: number
  language: string
  targets: string[]
  best: string | null
  summary: EvalTargetResult[]
  results: EvalTargetResult[]
}

export interface SystemInfo {
  os: string
  device: string
  cpu_percent: number
  cpu_count: number
  process: { rss_mb: number }
  ram: { total_mb: number; used_mb: number; free_mb: number; percent: number }
  gpus: { index: number; name: string; total_mb: number; used_mb: number; free_mb: number }[]
  note: string | null
}

export type Target = { kind: 'base'; model: string } | { kind: 'run'; run_id: string }
