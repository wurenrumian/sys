/**
 * 拥塞控制 —— 03_network/demo3 的浏览器版（流体近似，按 ms 推进）。
 * 队列深度 / 链路速率 = 排队延迟，所以「低队列」就是「低尾延迟」。
 */
import { percentile } from './format'

export const LINK_PKT_PER_MS = 100
export const RTT_MS = 1.0
export const BDP = LINK_PKT_PER_MS * RTT_MS
export const QUEUE_CAP = 500
export const SIM_MS = 4000
export const DCTCP_K = 20

export type CCId = 'reno' | 'dctcp' | 'bbr'

export interface CCInfo {
  id: CCId
  name: string
}
export const CCS: CCInfo[] = [
  { id: 'reno', name: 'Reno (丢包驱动)' },
  { id: 'dctcp', name: 'DCTCP (ECN驱动)' },
  { id: 'bbr', name: 'BBR式 (BDP驱动)' },
]

interface Flow {
  kind: CCId
  cwnd: number
  alpha: number
  bwEst: number
  minRtt: number
  t: number
}

const BBR_CYCLE = [1.25, 0.75, 1, 1, 1, 1, 1, 1]

function makeFlow(kind: CCId): Flow {
  return { kind, cwnd: 10, alpha: 0, bwEst: 1, minRtt: RTT_MS, t: 0 }
}

function onAck(f: Flow, marked: boolean, lost: boolean, k: number) {
  if (f.kind === 'bbr') return
  if (f.kind === 'reno') {
    if (lost) f.cwnd = Math.max(2, f.cwnd / 2)
    else f.cwnd += 1 / f.cwnd
    return
  }
  // dctcp
  void k
  const g = 0.0625
  f.alpha = (1 - g) * f.alpha + g * (marked ? 1 : 0)
  if (lost) f.cwnd = Math.max(2, f.cwnd / 2)
  else if (marked) f.cwnd = Math.max(2, f.cwnd * (1 - f.alpha / 2))
  else f.cwnd += 1 / f.cwnd
}

function onRound(f: Flow, deliveredPerMs: number, rtt: number) {
  f.bwEst = Math.max(f.bwEst * 0.9, deliveredPerMs)
  f.minRtt = Math.min(f.minRtt * 1.001, rtt)
  const gain = BBR_CYCLE[f.t % BBR_CYCLE.length]
  f.t++
  f.cwnd = Math.max(4, gain * f.bwEst * f.minRtt)
}

export interface CongResult {
  util: number
  qAvg: number
  qP99: number
  delayAvg: number
  delayP99: number
  drops: number
  jain: number
  qSeries: Float64Array
  utilSeries: Float64Array
  cwndSeries: Float64Array
}

export function simulateCC(kind: CCId, opts: { nFlows?: number; simMs?: number; k?: number } = {}): CongResult {
  const nFlows = opts.nFlows ?? 4
  const simMs = opts.simMs ?? SIM_MS
  const k = opts.k ?? DCTCP_K
  const flows = Array.from({ length: nFlows }, () => makeFlow(kind))
  let queue = 0
  let drops = 0
  const qSeries = new Float64Array(simMs)
  const utilSeries = new Float64Array(simMs)
  const cwndSeries = new Float64Array(simMs)

  for (let t = 0; t < simMs; t++) {
    let sumCwnd = 0
    for (const f of flows) sumCwnd += f.cwnd
    const offered = sumCwnd / RTT_MS
    const served = Math.min(offered + queue, LINK_PKT_PER_MS)
    queue = Math.max(0, queue + offered - LINK_PKT_PER_MS)
    const lost = queue > QUEUE_CAP
    if (lost) {
      drops += queue - QUEUE_CAP
      queue = QUEUE_CAP
    }
    const marked = queue > k
    for (const f of flows) {
      if (f.kind === 'bbr') {
        const rttNow = RTT_MS + queue / LINK_PKT_PER_MS
        onRound(f, served / nFlows, rttNow)
      } else {
        onAck(f, marked, lost, k)
      }
    }
    qSeries[t] = queue
    utilSeries[t] = served / LINK_PKT_PER_MS
    cwndSeries[t] = sumCwnd / nFlows
  }

  const start = Math.floor(simMs / 4)
  const q = qSeries.slice(start)
  const u = utilSeries.slice(start)
  const cw = cwndSeries.slice(start)
  const qSorted = Float64Array.from(q).sort()
  const last = Array.from({ length: nFlows }, (_, idx) => cwndSeries[simMs - 1] / nFlows + idx * 0)
  void last
  // 公平性用各流最终 cwnd（简化：流近似同速，Jain=1）
  const jain = 1.0
  let qSum = 0
  let uSum = 0
  for (let i = 0; i < q.length; i++) {
    qSum += q[i]
    uSum += u[i]
  }
  const qAvg = q.length ? qSum / q.length : 0
  const qP99 = percentile(qSorted, 99)
  void cw
  return {
    util: q.length ? uSum / q.length : 0,
    qAvg,
    qP99,
    delayAvg: RTT_MS + qAvg / LINK_PKT_PER_MS,
    delayP99: RTT_MS + qP99 / LINK_PKT_PER_MS,
    drops,
    jain,
    qSeries,
    utilSeries,
    cwndSeries,
  }
}
