/**
 * DCAI / 数据质量 —— 01_data_storage/demo3 的浏览器版。
 *
 * 关键设计：训练好的逻辑回归权重直接取自本机真实跑出的模型
 * （见 /tmp 提取脚本，训练集 60000 条、seed=1、AUC≈0.876），
 * 浏览器端只负责「生成数据 → 注入事故 → 打分 → 算 AUC / PSI」，
 * 所以 PSI 抓不到的那两类事故能被如实复现。
 */
import { mulberry32, gaussian, type Rng } from './rand'

export const N_FEAT = 12
export const W_TRUE = [1.4, 1.1, 0.9, 0.7, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
/** 由 Python 真实训练得到（train_lr, 250 epochs, lr=0.4） */
export const W_MODEL = [
  1.39144579, 1.10453031, 0.90923723, 0.71593951, 0.15621565, 0.16438211, 0.14396026,
  0.16456439, 0.16473093, 0.13695255, 0.16965366, 0.15178153,
]
export const B_MODEL = -1.00446207

export interface Dataset {
  /** 行优先，长度 n * N_FEAT */
  X: Float64Array
  y: Int8Array
  n: number
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/** 生成带真实信号的样本：前 4 个特征强、其余弱 */
export function makeData(seed: number, n: number, w = W_TRUE): Dataset {
  const rng = mulberry32(seed)
  const X = new Float64Array(n * N_FEAT)
  const y = new Int8Array(n)
  for (let i = 0; i < n; i++) {
    let logit = -1.0
    for (let j = 0; j < N_FEAT; j++) {
      const v = gaussian(rng)
      X[i * N_FEAT + j] = v
      logit += v * w[j]
    }
    y[i] = rng() < sigmoid(logit) ? 1 : 0
  }
  return { X, y, n }
}

function clone(X: Float64Array): Float64Array {
  return X.slice()
}

export type IncidentId = 'none' | 'missing' | 'shift' | 'skew' | 'stale'

export interface IncidentDef {
  id: IncidentId
  label: string
  blurb: string
}

export const INCIDENTS: IncidentDef[] = [
  { id: 'none', label: '无事故 (基线)', blurb: '健康流量' },
  { id: 'missing', label: '特征缺失: feat_0 全变默认值', blurb: '生产任务挂掉' },
  { id: 'shift', label: '口径变更: feat_1 平移+缩放', blurb: '上游改了口径' },
  { id: 'skew', label: '离在线不一致: feat_2 三成取到脏值', blurb: '部分流量命中旧版本' },
  { id: 'stale', label: '特征不新鲜: 全特征叠加噪声', blurb: '管道积压、特征延迟' },
]

/**
 * 注入事故，返回被污染特征的矩阵副本。
 * `ratio` 只对 missing / skew / stale 有意义（污染比例）。
 */
export function injectIncident(inc: IncidentId, X: Float64Array, n: number, seed: number, ratio = 0.3): Float64Array {
  const B = clone(X)
  const rng = mulberry32(seed * 131 + 7)
  if (inc === 'missing') {
    for (let i = 0; i < n; i++) B[i * N_FEAT] = 0
  } else if (inc === 'shift') {
    for (let i = 0; i < n; i++) B[i * N_FEAT + 1] = B[i * N_FEAT + 1] * 2.5 + 1.8
  } else if (inc === 'skew') {
    for (let i = 0; i < n; i++) {
      if (rng() < ratio) B[i * N_FEAT + 2] = gaussian(rng)
    }
  } else if (inc === 'stale') {
    const sigma = 0.35
    for (let i = 0; i < n * N_FEAT; i++) B[i] += gaussian(rng) * sigma
  }
  return B
}

/** 用模型给每个样本打分：score = X·w + b */
export function score(X: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let z = B_MODEL
    for (let j = 0; j < N_FEAT; j++) z += X[i * N_FEAT + j] * W_MODEL[j]
    out[i] = z
  }
  return out
}

/** 取第 j 列 */
export function column(X: Float64Array, n: number, j: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = X[i * N_FEAT + j]
  return out
}

/** Population Stability Index：期望分布 = 训练期，实际分布 = 线上 */
export function psi(expect: Float64Array, actual: Float64Array, bins = 10): number {
  const sorted = Float64Array.from(expect).sort()
  const edges = new Float64Array(bins + 1)
  for (let b = 0; b <= bins; b++) {
    const pos = (sorted.length - 1) * (b / bins)
    const lo = Math.floor(pos)
    const hi = Math.min(lo + 1, sorted.length - 1)
    edges[b] = sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
  }
  edges[0] = -Infinity
  edges[bins] = Infinity

  const e = new Float64Array(bins)
  const a = new Float64Array(bins)
  const hist = (arr: Float64Array, out: Float64Array) => {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]
      let lo = 0
      let hi = bins - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (v < edges[mid + 1]) hi = mid
        else lo = mid + 1
      }
      out[lo]++
    }
    for (let b = 0; b < bins; b++) out[b] /= arr.length
  }
  hist(expect, e)
  hist(actual, a)

  const eps = 1e-6
  let s = 0
  for (let b = 0; b < bins; b++) {
    const ee = Math.max(e[b], eps)
    const aa = Math.max(a[b], eps)
    s += (aa - ee) * Math.log(aa / ee)
  }
  return s
}

/** 秩和法 AUC，处理并列值取平均秩 */
export function auc(y: Int8Array, s: Float64Array): number {
  const n = s.length
  const idx = Array.from({ length: n }, (_, i) => i).sort((p, q) => s[p] - s[q])
  const rank = new Float64Array(n)
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && s[idx[j + 1]] === s[idx[i]]) j++
    const avg = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) rank[idx[k]] = avg
    i = j + 1
  }
  let n1 = 0
  let sumPos = 0
  for (let k = 0; k < n; k++) {
    if (y[k] === 1) {
      n1++
      sumPos += rank[k]
    }
  }
  const n0 = n - n1
  if (n1 === 0 || n0 === 0) return 0.5
  return (sumPos - (n1 * (n1 + 1)) / 2) / (n1 * n0)
}

export function maxPsi(
  Xtr: Float64Array,
  nTr: number,
  Xbad: Float64Array,
  nBad: number,
): { value: number; feat: number } {
  let best = -1
  let bestFeat = 0
  for (let j = 0; j < N_FEAT; j++) {
    const v = psi(column(Xtr, nTr, j), column(Xbad, nBad, j))
    if (v > best) {
      best = v
      bestFeat = j
    }
  }
  return { value: best, feat: bestFeat }
}
