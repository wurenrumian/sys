/**
 * AllReduce 通信模型 —— 03_network/demo2 的浏览器版。
 * T = 通信轮数 × 单跳延迟 + 传输字节 / 带宽（alpha-beta 模型）。
 */

export interface Link {
  id: string
  name: string
  bw: number // GB/s
  lat: number // 单跳延迟 µs
}

export const LINKS: Link[] = [
  { id: 'nvlink', name: 'NVLink (机内)', bw: 450, lat: 1.0 },
  { id: 'ib', name: 'InfiniBand (机间)', bw: 25, lat: 2.5 },
  { id: 'roce', name: 'RoCE 以太网', bw: 12.5, lat: 5.0 },
]

export function linkById(id: string): Link {
  return LINKS.find((l) => l.id === id) ?? LINKS[1]
}

/** 朴素中心汇聚：所有卡发给 rank0，再广播回去。单点带宽瓶颈。 */
export function naive(n: number, D: number, bw: number, lat: number): number {
  return 2 * lat * 1e-6 + (2 * (n - 1) * D) / (bw * 1e9)
}

/** Ring：reduce-scatter + all-gather，各 n-1 步，共 2(n-1) 步。 */
export function ring(n: number, D: number, bw: number, lat: number): number {
  const steps = 2 * (n - 1)
  return steps * lat * 1e-6 + (steps * (D / n)) / (bw * 1e9)
}

/** 二项树：2*log2(n) 步，但每步传完整 D。 */
export function tree(n: number, D: number, bw: number, lat: number): number {
  const steps = 2 * Math.ceil(Math.log2(Math.max(2, n)))
  return steps * lat * 1e-6 + (steps * D) / (bw * 1e9)
}

/** 递归折半-加倍：log n 轮 + Ring 级传输量（理想模型下同时最优）。 */
export function halvingDoubling(n: number, D: number, bw: number, lat: number): number {
  const steps = 2 * Math.ceil(Math.log2(Math.max(2, n)))
  const total = (2 * D * (n - 1)) / n
  return steps * lat * 1e-6 + total / (bw * 1e9)
}

export interface Algo {
  id: string
  name: string
  fn: (n: number, D: number, bw: number, lat: number) => number
}

export const ALGOS: Algo[] = [
  { id: 'naive', name: 'Naive(中心汇聚)', fn: naive },
  { id: 'ring', name: 'Ring', fn: ring },
  { id: 'tree', name: '二项树', fn: tree },
  { id: 'hd', name: '递归折半HD', fn: halvingDoubling },
]

/** (n-1) 步、每步 D/n，结束后每卡持有 1/n 的规约结果 */
export function reduceScatter(n: number, D: number, bw: number, lat: number): number {
  return (n - 1) * lat * 1e-6 + ((n - 1) * (D / n)) / (bw * 1e9)
}

export interface HierResult {
  flat: number
  hier: number
  rs: number
  inter: number
  ag: number
}

/** 分层 AllReduce：机内 reduce-scatter → 机间 AllReduce(仅 D/G) → 机内 all-gather */
export function hierarchical(opts: {
  machines: number
  gpusPerMachine: number
  D: number
  intra: Link
  inter: Link
}): HierResult {
  const { machines: M, gpusPerMachine: G, D, intra, inter } = opts
  const flat = ring(G * M, D, inter.bw, inter.lat)
  const rs = reduceScatter(G, D, intra.bw, intra.lat)
  const tInter = ring(M, D / G, inter.bw, inter.lat)
  const ag = reduceScatter(G, D, intra.bw, intra.lat)
  return { flat, hier: rs + tInter + ag, rs, inter: tInter, ag }
}
