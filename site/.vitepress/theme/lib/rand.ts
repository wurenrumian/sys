/** 可播种的随机数工具：给浏览器内的模拟提供确定性、可复现的随机流。 */

export type Rng = () => number

/** mulberry32：小而快，返回 [0,1) */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return function () {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 指数分布，均值 = mean */
export function expo(rng: Rng, mean: number): number {
  return -Math.log(1 - rng()) * mean
}

/** 标准正态（Box-Muller） */
export function gaussian(rng: Rng): number {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/**
 * Zipf 分布采样（与 numpy.random.zipf 行为对齐）。
 *
 * numpy 的 zipf 支持无界：以 a=1.15、上界 1e6 为例，约 11.5% 的样例会落在
 * 1e6 之外，取模后近似均匀地摊到整个 key 空间。若只做截断采样，头部会过度集中
 * （最热 1% 承接 ~88% 而非脚本里的 ~67%）。所以这里显式建模那条尾巴。
 *
 * @param a  指数（越大越集中）
 * @param m  support 上界（取值 1..m）
 * @param n  采样条数
 * @returns 长度为 n 的数组，元素在 1..m
 */
export function zipfSample(rng: Rng, a: number, m: number, n: number): Float64Array {
  // 预计算累积分布 + 二分查找，比每次重算快很多
  const cdf = new Float64Array(m)
  let sum = 0
  for (let k = 1; k <= m; k++) {
    sum += 1 / Math.pow(k, a)
    cdf[k - 1] = sum
  }
  // 尾巴质量：∫_m^∞ x^-a dx = m^(1-a)/(a-1)，相对总质量的比例
  const tailIntegral = a > 1 ? Math.pow(m, 1 - a) / (a - 1) : 0
  const pTail = tailIntegral / (sum + tailIntegral)
  for (let i = 0; i < m; i++) cdf[i] /= sum

  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (pTail > 0 && rng() < pTail) {
      out[i] = 1 + Math.floor(rng() * m) // 尾巴取模后近似均匀
      continue
    }
    const u = rng()
    let lo = 0
    let hi = m - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cdf[mid] < u) lo = mid + 1
      else hi = mid
    }
    out[i] = lo + 1
  }
  return out
}

export function randInt(rng: Rng, n: number): number {
  return Math.floor(rng() * n)
}

/** 0..n 之间取一个整数（含 n） */
export function randIntInclusive(rng: Rng, n: number): number {
  return Math.floor(rng() * (n + 1))
}
