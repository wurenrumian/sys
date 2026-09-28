/**
 * 多级存储 + Embedding Cache —— 01_data_storage/demo2 的浏览器版。
 * 长尾（Zipf）访问下，比较 LRU / LFU，以及"提命中率"与"让某层更快"哪个更值。
 */
import { mulberry32, zipfSample, type Rng } from './rand'

export const N_KEYS = 1_000_000
export const N_REQ = 400_000
export const ZIPF_A = 1.15

export interface Tier {
  name: string
  lat: number // µs
  cost: number // 相对单位成本
}
export const TIERS: Tier[] = [
  { name: 'GPU HBM', lat: 0.2, cost: 100 },
  { name: '本机 DRAM', lat: 2.0, cost: 10 },
  { name: '远程 DRAM/RDMA', lat: 10.0, cost: 8 },
  { name: '本地 NVMe', lat: 100.0, cost: 1 },
  { name: '对象存储', lat: 10_000.0, cost: 0.05 },
]

export interface Trace {
  keys: Int32Array
  nKeys: number
  nReq: number
  distinct: number
  top1pct: number
}

/**
 * 生成 Zipf 访问序列。
 * churnEveryMs 不为空时，每隔这么多请求把 key 空间整体平移一次
 * （模拟新内容爆火、旧热点过气；分布形状不变，变的是"谁是热点"）。
 */
export function genTrace(opts: {
  seed?: number
  nReq?: number
  nKeys?: number
  a?: number
  churnEvery?: number | null
} = {}): Trace {
  const { seed = 7, nReq = N_REQ, nKeys = N_KEYS, a = ZIPF_A, churnEvery = null } = opts
  const rng = mulberry32(seed + 9)
  const raw = zipfSample(rng, a, nKeys, nReq)
  const keys = new Int32Array(nReq)
  for (let i = 0; i < nReq; i++) keys[i] = Math.floor(raw[i]) % nKeys

  if (churnEvery) {
    for (let start = churnEvery; start < nReq; start += churnEvery) {
      const offset = 1 + Math.floor(rng() * (nKeys - 1))
      const end = Math.min(start + churnEvery, nReq)
      for (let i = start; i < end; i++) keys[i] = (keys[i] + offset) % nKeys
    }
  }

  // 最热的 1% **访问过的** key 承接了多少请求（与脚本口径一致）
  const counts = new Int32Array(nKeys)
  let distinct = 0
  for (let i = 0; i < nReq; i++) {
    if (counts[keys[i]] === 0) distinct++
    counts[keys[i]]++
  }
  const sorted = Array.from(counts).sort((p, q) => q - p)
  const top = Math.max(1, Math.floor(distinct / 100))
  let s = 0
  for (let i = 0; i < top; i++) s += sorted[i]
  return { keys, nKeys, nReq, distinct, top1pct: s / nReq }
}

// ---------------------------------------------------------------- 缓存策略
export interface Policy {
  name: string
  get(k: number): boolean
}

class LRU implements Policy {
  name = 'LRU'
  cap: number
  private map = new Map<number, true>()
  constructor(cap: number) {
    this.cap = cap
  }
  get(k: number): boolean {
    if (this.map.has(k)) {
      this.map.delete(k)
      this.map.set(k, true)
      return true
    }
    if (this.map.size >= this.cap) {
      const oldest = this.map.keys().next().value
      if (oldest !== undefined) this.map.delete(oldest)
    }
    this.map.set(k, true)
    return false
  }
}

class LFU implements Policy {
  name = 'LFU'
  cap: number
  private live = new Set<number>()
  private keys: number[] = []
  private freq = new Map<number, number>()
  private n = 0
  private rng: Rng
  private SAMPLE = 8
  constructor(cap: number, seed = 11) {
    this.cap = cap
    this.rng = mulberry32(seed)
  }
  get(k: number): boolean {
    this.n++
    this.freq.set(k, (this.freq.get(k) ?? 0) + 1)
    if (this.n % 200_000 === 0) {
      for (const [key, v] of this.freq) this.freq.set(key, v >> 1)
    }
    if (this.live.has(k)) return true
    if (this.live.size >= this.cap) this.evict()
    this.live.add(k)
    this.keys.push(k)
    return false
  }
  private evict() {
    let best = -1
    let bestF = Infinity
    let tries = 0
    while (this.keys.length && tries < this.SAMPLE * 4) {
      const i = Math.floor(this.rng() * this.keys.length)
      const cand = this.keys[i]
      if (!this.live.has(cand)) {
        this.keys[i] = this.keys[this.keys.length - 1]
        this.keys.pop()
        continue
      }
      const f = this.freq.get(cand) ?? 0
      if (f < bestF) {
        best = cand
        bestF = f
      }
      tries++
      if (tries >= this.SAMPLE) break
    }
    if (best >= 0) this.live.delete(best)
  }
}

export function makePolicy(kind: 'lru' | 'lfu', cap: number): Policy {
  return kind === 'lru' ? new LRU(cap) : new LFU(cap)
}

export function hitRate(trace: Trace, kind: 'lru' | 'lfu', cap: number): number {
  const p = makePolicy(kind, cap)
  let hits = 0
  const keys = trace.keys
  for (let i = 0; i < trace.nReq; i++) if (p.get(keys[i])) hits++
  return hits / trace.nReq
}

// ---------------------------------------------------------------- 端到端有效延迟
/**
 * hitRates[i] = 在第 i 层命中的请求占**总流量**的绝对比例；
 * 剩下的 1-sum 是穿透率，落到最后一层兜底。
 */
export function effectiveLatency(hitRates: number[], tiers = TIERS): number {
  let lat = 0
  for (let i = 0; i < hitRates.length && i < tiers.length; i++) lat += hitRates[i] * tiers[i].lat
  const sum = hitRates.reduce((a, b) => a + b, 0)
  lat += Math.max(0, 1 - sum) * tiers[tiers.length - 1].lat
  return lat
}

export const TIER_DIST = [0.005, 0.05, 0.15, 0.3, 1.0]
export function tierCost(dist: number[], tiers = TIERS): number {
  let total = 0
  for (let i = 0; i < dist.length; i++) total += dist[i] * tiers[i].cost
  return total
}
