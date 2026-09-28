/**
 * 动态批处理 —— 02_compile/demo4 的浏览器版。
 * 单 GPU worker 的离散事件模拟：攒够多少 / 等多久才发一批。
 */
import { mulberry32, expo } from './rand'

export const SIM_MS = 60_000
export const BASE_QPS = 800
export const SPIKE_AT: [number, number] = [30_000, 34_000]
const EPS = 1e-9

export function batchTimeMs(bs: number): number {
  return 2.0 + 0.012 * bs
}

export function genArrivals(seed = 0, baseQps = BASE_QPS): Float64Array {
  const rng = mulberry32(seed + 5)
  const times: number[] = []
  let t = 0
  while (t < SIM_MS) {
    let qps = baseQps * (1 + 0.4 * Math.sin((2 * Math.PI * t) / 20_000))
    if (t >= SPIKE_AT[0] && t < SPIKE_AT[1]) qps *= 3
    t += expo(rng, 1000 / qps)
    if (t < SIM_MS) times.push(t)
  }
  return Float64Array.from(times)
}

export type BatchPolicy = 'fixed_bs' | 'fixed_timeout' | 'adaptive'

export interface BatchConfig {
  policy: BatchPolicy
  fixedBs?: number
  maxBs?: number
  timeoutMs?: number
}

export interface BatchResult {
  p50: number
  p99: number
  p999: number
  max: number
  avgBs: number
  nBatch: number
  gpuUtil: number
  latencies: Float64Array
  batches: Array<{ t0: number; t1: number; bs: number }>
}

export function simulateBatch(arrivals: Float64Array, cfg: BatchConfig): BatchResult {
  const maxBs = cfg.maxBs ?? 256
  const fixedBs = cfg.fixedBs ?? 64
  const timeoutMs = cfg.timeoutMs ?? 5
  const n = arrivals.length
  const q = new Int32Array(n)
  let head = 0
  let tail = 0
  let i = 0
  let now = 0
  let gpuBusy = 0
  const latencies = new Float64Array(n)
  const batches: Array<{ t0: number; t1: number; bs: number }> = []

  while (i < n || tail > head) {
    while (i < n && arrivals[i] <= now) {
      q[tail++] = i
      i++
    }
    if (tail === head) {
      now = arrivals[i]
      continue
    }
    const len = tail - head
    const oldestWait = now - arrivals[q[head]]

    let bs: number
    if (cfg.policy === 'fixed_bs') {
      if (len < fixedBs && i < n) {
        now = arrivals[i]
        continue
      }
      bs = Math.min(len, fixedBs)
    } else if (cfg.policy === 'fixed_timeout') {
      if (len < maxBs && oldestWait < timeoutMs - EPS && i < n) {
        now = Math.min(arrivals[i], arrivals[q[head]] + timeoutMs)
        continue
      }
      bs = Math.min(len, maxBs)
    } else {
      if (len >= maxBs) {
        bs = maxBs
      } else {
        const dynTimeout = 0.5 + 4.0 * (len / maxBs)
        if (oldestWait < dynTimeout - EPS && i < n) {
          now = Math.min(arrivals[i], arrivals[q[head]] + dynTimeout)
          continue
        }
        bs = len
      }
    }

    const dur = batchTimeMs(bs)
    const t0 = now
    gpuBusy += dur
    const finish = now + dur
    for (let k = 0; k < bs; k++) latencies[q[head + k]] = finish - arrivals[q[head + k]]
    batches.push({ t0, t1: finish, bs })
    head += bs
    now = finish
  }

  const sorted = Float64Array.from(latencies).sort()
  const pct = (p: number) => {
    const m = sorted.length
    if (m === 0) return 0
    const pos = (m - 1) * (p / 100)
    const lo = Math.floor(pos)
    const hi = Math.min(lo + 1, m - 1)
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
  }
  let sumBs = 0
  for (const b of batches) sumBs += b.bs
  return {
    p50: pct(50),
    p99: pct(99),
    p999: pct(99.9),
    max: sorted.length ? sorted[sorted.length - 1] : 0,
    avgBs: batches.length ? sumBs / batches.length : 0,
    nBatch: batches.length,
    gpuUtil: gpuBusy / Math.max(SIM_MS, now),
    latencies,
    batches,
  }
}
