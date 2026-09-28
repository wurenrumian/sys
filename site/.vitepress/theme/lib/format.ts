/** 数值格式化助手，统一各 demo 的展示口径。 */

export function fmtUs(us: number, digits = 1): string {
  if (!isFinite(us)) return '∞'
  if (us >= 1e6) return (us / 1e6).toFixed(2) + ' s'
  if (us >= 1e3) return (us / 1e3).toFixed(digits) + ' ms'
  if (us >= 1) return us.toFixed(digits) + ' µs'
  return us.toFixed(3) + ' µs'
}

export function fmtMs(ms: number, digits = 2): string {
  return fmtUs(ms * 1e3, digits)
}

export function fmtBytes(bytes: number): string {
  if (bytes >= 1e15) return (bytes / 1e15).toFixed(1) + ' PB'
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(1) + ' TB'
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB'
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB'
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB'
  return bytes + ' B'
}

export function fmtPct(x: number, digits = 1): string {
  return (x * 100).toFixed(digits) + '%'
}

export function fmtInt(x: number): string {
  return Math.round(x).toLocaleString('en-US')
}

/** numpy 风格的线性插值百分位 */
export function percentile(sortedAsc: ArrayLike<number>, q: number): number {
  const n = sortedAsc.length
  if (n === 0) return 0
  if (n === 1) return sortedAsc[0]
  const pos = (n - 1) * (q / 100)
  const lo = Math.floor(pos)
  const frac = pos - lo
  const hi = Math.min(lo + 1, n - 1)
  return sortedAsc[lo] + frac * (sortedAsc[hi] - sortedAsc[lo])
}
