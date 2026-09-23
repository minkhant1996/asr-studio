import { useEffect, useState } from 'react'
import { api } from '../api'
import type { SystemInfo } from '../types'

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`

export default function SystemStats({ compact }: { compact?: boolean }) {
  const [s, setS] = useState<SystemInfo | null>(null)
  useEffect(() => {
    const load = () => api.system().then(setS).catch(() => setS(null))
    load()
    const t = setInterval(load, 4000)
    return () => clearInterval(t)
  }, [])
  if (!s) return null
  const gpu = s.gpus[0]
  if (compact) {
    return (
      <span className="small" title={`${s.os} · ${s.cpu_count} CPU threads${s.note ? ` · ${s.note}` : ''}`}>
        · {s.device.toUpperCase()} · RAM {gb(s.ram.used_mb)}/{gb(s.ram.total_mb)}
        {gpu ? ` · VRAM ${gb(gpu.used_mb)}/${gb(gpu.total_mb)}` : ''}
      </span>
    )
  }
  const Meter = ({ label, used, total, extra }: { label: string; used: number; total: number; extra?: string }) => {
    const pct = total ? Math.min(100, (used / total) * 100) : 0
    return (
      <div className="metric" style={{ minWidth: 200 }}>
        <div className="k">{label}</div>
        <div className="v" style={{ fontSize: 16 }}>
          {gb(used)} <span className="small">/ {gb(total)}</span>
        </div>
        <div className="bar">
          <div style={{ width: `${pct}%`, background: pct > 90 ? 'var(--bad)' : pct > 75 ? '#ffb454' : 'var(--accent)' }} />
        </div>
        {extra && <div className="small">{extra}</div>}
      </div>
    )
  }
  return (
    <section className="panel">
      <h2>Hardware · {s.os}</h2>
      <div className="metrics" style={{ flexWrap: 'wrap' }}>
        <div className="metric">
          <div className="k">compute device</div>
          <div className="v">{s.device.toUpperCase()}</div>
          <div className="small">{s.cpu_count} CPU threads · {s.cpu_percent.toFixed(0)}% busy</div>
        </div>
        <Meter label="system RAM" used={s.ram.used_mb} total={s.ram.total_mb} />
        <Meter label="backend process" used={s.process.rss_mb} total={s.ram.total_mb} />
        {s.gpus.map((g) => (
          <Meter key={g.index} label={`VRAM · ${g.name}`} used={g.used_mb} total={g.total_mb} />
        ))}
        {s.gpus.length === 0 && (
          <div className="metric">
            <div className="k">GPU</div>
            <div className="v" style={{ fontSize: 16 }}>none detected</div>
          </div>
        )}
      </div>
      {s.note && <div className="small" style={{ color: '#ffb454' }}>⚠ {s.note}</div>}
    </section>
  )
}
