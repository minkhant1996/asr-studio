import { useState } from 'react'

/** Single-series training loss. One series, so no legend: the title names it. */
export default function LossChart({ points, title = 'Training loss' }: { points: { step: number; loss: number }[]; title?: string }) {
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null)
  if (points.length < 2) return <div className="small">Loss appears once a few steps have run.</div>
  const w = 640
  const h = 200
  const pad = { l: 46, r: 12, t: 10, b: 26 }
  const xs = points.map((p) => p.step)
  const ys = points.map((p) => p.loss)
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs)
  const y0 = Math.min(...ys)
  const y1 = Math.max(...ys)
  const sx = (v: number) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r)
  const sy = (v: number) => h - pad.b - ((v - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b)
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(' ')
  const yticks = [y0, (y0 + y1) / 2, y1]
  const xticks = [x0, Math.round((x0 + x1) / 2), x1]
  const last = points[points.length - 1]
  const first = points[0]
  return (
    <div>
      <div className="small" style={{ marginBottom: 4 }}>
        {title} · {first.loss.toFixed(3)} → <b style={{ color: last.loss <= first.loss ? 'var(--ok)' : 'var(--bad)' }}>{last.loss.toFixed(3)}</b>
      </div>
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={title} onMouseLeave={() => setHover(null)}>
        <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="#8b93a5" strokeWidth={1} />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="#8b93a5" strokeWidth={1} />
        {yticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} y1={sy(t)} x2={w - pad.r} y2={sy(t)} stroke="#262b36" strokeWidth={1} />
            <text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        {xticks.map((t) => (
          <text key={t} x={sx(t)} y={h - pad.b + 16} textAnchor="middle">{t}</text>
        ))}
        <path d={d} fill="none" stroke="#6c8cff" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.step)}
            cy={sy(p.loss)}
            r={points.length > 80 ? 2 : 4}
            fill="#6c8cff"
            stroke="#0f1115"
            strokeWidth={1}
            onMouseMove={(e) => setHover({ x: e.clientX, y: e.clientY, text: `step ${p.step}: loss ${p.loss.toFixed(4)}` })}
          />
        ))}
        <text x={(w - pad.l) / 2 + pad.l} y={h - 2} textAnchor="middle">step</text>
      </svg>
      {hover && (
        <div className="tt" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          {hover.text}
        </div>
      )}
    </div>
  )
}
