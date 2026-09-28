/**
 * 显存碎片 —— 02_compile/demo3 的浏览器版。
 * 四种分配器跑同一段不定长分配/释放序列，看谁先 OOM、碎片从哪来。
 */
import { mulberry32, type Rng } from './rand'

export type AllocId = 'first' | 'best' | 'buddy' | 'paged'

export interface AllocStats {
  used: number
  freeTotal: number
  largest: number
  frag: number
  nfree: number
  internal: number
}

export interface Allocator {
  name: string
  malloc(id: number, size: number): boolean
  free(id: number): void
  stats(): AllocStats
  /** 用于可视化：返回 [起址, 大小, 是否占用] 的区间列表 */
  intervals(total: number): Array<[number, number, boolean]>
  oom: number
}

// ---------------------------------------------------------------- First/Best-Fit
export class FreeListAllocator implements Allocator {
  total: number
  policy: 'first' | 'best'
  name: string
  private holes: Array<[number, number]> = []
  private alloc = new Map<number, [number, number]>()
  oom = 0

  constructor(total: number, policy: 'first' | 'best') {
    this.total = total
    this.policy = policy
    this.holes = [[0, total]]
    this.name = policy === 'first' ? 'First-Fit(连续)' : 'Best-Fit(连续)'
  }

  malloc(id: number, size: number): boolean {
    let cand = -1
    for (let i = 0; i < this.holes.length; i++) {
      if (this.holes[i][1] >= size) {
        if (this.policy === 'first') {
          cand = i
          break
        }
        if (cand < 0 || this.holes[i][1] < this.holes[cand][1]) cand = i
      }
    }
    if (cand < 0) {
      this.oom++
      return false
    }
    const [off, sz] = this.holes[cand]
    this.alloc.set(id, [off, size])
    if (sz === size) this.holes.splice(cand, 1)
    else this.holes[cand] = [off + size, sz - size]
    return true
  }

  free(id: number): void {
    const a = this.alloc.get(id)
    if (!a) return
    this.alloc.delete(id)
    this.holes.push(a)
    this.holes.sort((p, q) => p[0] - q[0])
    const merged: Array<[number, number]> = [this.holes[0]]
    for (let i = 1; i < this.holes.length; i++) {
      const [o, s] = this.holes[i]
      const [po, ps] = merged[merged.length - 1]
      if (po + ps === o) merged[merged.length - 1] = [po, ps + s]
      else merged.push([o, s])
    }
    this.holes = merged
  }

  stats(): AllocStats {
    let used = 0
    for (const a of this.alloc.values()) used += a[1]
    const freeTotal = this.total - used
    let largest = 0
    for (const [, s] of this.holes) largest = Math.max(largest, s)
    const frag = freeTotal > 0 ? 1 - largest / freeTotal : 0
    return { used, freeTotal, largest, frag, nfree: this.holes.length, internal: 0 }
  }

  intervals(): Array<[number, number, boolean]> {
    const usedRanges: Array<[number, number]> = []
    for (const [, a] of this.alloc) usedRanges.push(a)
    usedRanges.sort((p, q) => p[0] - q[0])
    const out: Array<[number, number, boolean]> = []
    let cur = 0
    for (const [off, sz] of usedRanges) {
      if (off > cur) out.push([cur, off - cur, false])
      out.push([off, sz, true])
      cur = off + sz
    }
    if (cur < this.total) out.push([cur, this.total - cur, false])
    return out
  }
}

// ---------------------------------------------------------------- Buddy
export class BuddyAllocator implements Allocator {
  order: number
  total: number
  name = 'Buddy(2的幂)'
  private holes: Array<Set<number>>
  private alloc = new Map<number, [number, number, number]>() // off, order, size
  oom = 0

  constructor(total: number) {
    this.order = Math.ceil(Math.log2(total))
    this.total = 2 ** this.order
    this.holes = Array.from({ length: this.order + 1 }, () => new Set<number>())
    this.holes[this.order].add(0)
  }

  private orderOf(size: number): number {
    return Math.max(0, Math.ceil(Math.log2(Math.max(1, size))))
  }

  malloc(id: number, size: number): boolean {
    const need = this.orderOf(size)
    let o = need
    while (o <= this.order && this.holes[o].size === 0) o++
    if (o > this.order) {
      this.oom++
      return false
    }
    const it = this.holes[o].values().next()
    const off = it.value
    this.holes[o].delete(off)
    while (o > need) {
      o--
      this.holes[o].add(off + 2 ** o)
    }
    this.alloc.set(id, [off, need, size])
    return true
  }

  free(id: number): void {
    const a = this.alloc.get(id)
    if (!a) return
    this.alloc.delete(id)
    let [off, o] = a
    while (o < this.order) {
      const buddy = off ^ (1 << o)
      if (this.holes[o].has(buddy)) {
        this.holes[o].delete(buddy)
        off = Math.min(off, buddy)
        o++
      } else break
    }
    this.holes[o].add(off)
  }

  stats(): AllocStats {
    let usedReal = 0
    let usedReq = 0
    for (const a of this.alloc.values()) {
      usedReal += 2 ** a[1] // a = [off, order, size]
      usedReq += a[2]
    }
    const freeTotal = this.total - usedReal
    let largest = 0
    let nfree = 0
    for (let o = 0; o <= this.order; o++) {
      nfree += this.holes[o].size
      if (this.holes[o].size) largest = Math.max(largest, 2 ** o)
    }
    const frag = freeTotal > 0 ? 1 - largest / freeTotal : 0
    const internal = usedReal ? 1 - usedReq / usedReal : 0
    return { used: usedReal, freeTotal, largest, frag, nfree, internal }
  }

  intervals(): Array<[number, number, boolean]> {
    const usedRanges: Array<[number, number]> = []
    for (const [, [off, o]] of this.alloc) usedRanges.push([off, 2 ** o])
    usedRanges.sort((p, q) => p[0] - q[0])
    const out: Array<[number, number, boolean]> = []
    let cur = 0
    for (const [off, sz] of usedRanges) {
      if (off > cur) out.push([cur, off - cur, false])
      out.push([off, sz, true])
      cur = off + sz
    }
    if (cur < this.total) out.push([cur, this.total - cur, false])
    return out
  }
}

// ---------------------------------------------------------------- Paged
export class PagedAllocator implements Allocator {
  page: number
  npages: number
  total: number
  name: string
  private holes: number[]
  private alloc = new Map<number, [number[], number]>()
  oom = 0

  constructor(total: number, page = 4) {
    this.page = page
    this.npages = Math.floor(total / page)
    this.total = this.npages * page
    this.holes = Array.from({ length: this.npages }, (_, i) => i)
    const label = page >= 1 ? `${page}MB` : `${Math.round(page * 1024)}KB`
    this.name = `Paged(${label}/页)`
  }

  malloc(id: number, size: number): boolean {
    const need = Math.ceil(size / this.page)
    if (need > this.holes.length) {
      this.oom++
      return false
    }
    const pages: number[] = []
    for (let k = 0; k < need; k++) pages.push(this.holes.pop()!)
    this.alloc.set(id, [pages, size])
    return true
  }

  free(id: number): void {
    const a = this.alloc.get(id)
    if (!a) return
    this.alloc.delete(id)
    for (const p of a[0]) this.holes.push(p)
  }

  stats(): AllocStats {
    let usedReal = 0
    let usedReq = 0
    for (const [, [pages, s]] of this.alloc) {
      usedReal += pages.length * this.page
      usedReq += s
    }
    const freeTotal = this.total - usedReal
    const internal = usedReal ? 1 - usedReq / usedReal : 0
    return { used: usedReal, freeTotal, largest: freeTotal, frag: 0, nfree: this.holes.length, internal }
  }

  intervals(): Array<[number, number, boolean]> {
    // 展示每个物理页是否被占用
    const used = new Set<number>()
    for (const [, [pages]] of this.alloc) for (const p of pages) used.add(p)
    const out: Array<[number, number, boolean]> = []
    for (let p = 0; p < this.npages; p++) out.push([p * this.page, this.page, used.has(p)])
    return out
  }
}

export function makeAllocator(id: AllocId, total: number, page = 4): Allocator {
  if (id === 'first') return new FreeListAllocator(total, 'first')
  if (id === 'best') return new FreeListAllocator(total, 'best')
  if (id === 'buddy') return new BuddyAllocator(total)
  return new PagedAllocator(total, page)
}

// ---------------------------------------------------------------- 负载
export type WorkMode = 'skewed' | 'uniform'
export type Op = ['m' | 'f', number, number]

function pareto(rng: Rng, a: number): number {
  return Math.pow(1 - rng(), -1 / a) - 1
}

export function genWorkload(opts: {
  seed?: number
  mode?: WorkMode
  watermark?: number
  gpuMb?: number
  nOps?: number
}): Op[] {
  const { seed = 0, mode = 'skewed', watermark = 0.65, gpuMb = 4096, nOps = 20000 } = opts
  const rng = mulberry32(seed + 3)
  const cap = gpuMb * watermark
  const ops: Op[] = []
  const live: Array<[number, number]> = []
  let nid = 0
  let liveMb = 0
  for (let k = 0; k < nOps; k++) {
    let size: number
    if (mode === 'skewed') {
      size = Math.min(400, Math.max(1, pareto(rng, 1.2) * 6 + 1))
    } else {
      size = 1 + rng() * 59
    }
    if (live.length && (liveMb + size > cap || rng() >= 0.6)) {
      const i = Math.floor(rng() * live.length)
      const [aid, sz] = live[i]
      live[i] = live[live.length - 1]
      live.pop()
      liveMb -= sz
      ops.push(['f', aid, 0])
    } else {
      ops.push(['m', nid, size])
      live.push([nid, size])
      liveMb += size
      nid++
    }
  }
  return ops
}

export interface RunResult {
  oom: number
  peakUsed: number
  fragAvg: number
  fragEnd: number
  nfree: number
  internal: number
}

export function run(alloc: Allocator, ops: Op[]): RunResult {
  let peakUsed = 0
  let fragSum = 0
  let samples = 0
  for (let i = 0; i < ops.length; i++) {
    const [kind, aid, size] = ops[i]
    if (kind === 'm') alloc.malloc(aid, size)
    else alloc.free(aid)
    if (i % 200 === 0) {
      const st = alloc.stats()
      peakUsed = Math.max(peakUsed, st.used)
      fragSum += st.frag
      samples++
    }
  }
  const st = alloc.stats()
  return {
    oom: alloc.oom,
    peakUsed,
    fragAvg: samples ? fragSum / samples : 0,
    fragEnd: st.frag,
    nfree: st.nfree,
    internal: st.internal,
  }
}
