/**
 * RPC 小包 vs 集合通信大包 —— 单链路调度的离散事件模型（浏览器版）
 *
 * 这是 03_network/demo1_traffic_conflict.py 的忠实移植：
 * 同样的 store-and-forward / 不可抢占建模、同样的调度策略与带宽账。
 * 唯一差异是随机数发生器（脚本用 numpy PCG，这里用 mulberry32），
 * 所以具体数值有统计波动，但量级关系和结论完全一致。
 *
 * 关键建模点：一个包一旦开始发送就【不可抢占】。大包的发送时长
 * （4MB @ 100Gbps = 320µs）直接决定了后来者的最坏等待时间，这就是队头阻塞。
 */

export const LINK_GBPS = 100;
export const DEFAULT_RPC_QPS = 20_000;
export const DEFAULT_RPC_BYTES = 512;
export const DEFAULT_BULK_MB = 4;
export const DEFAULT_PERIOD_MS = 2;
export const SIM_MS = 2000;

export type Policy = 'fifo' | 'priority' | 'wrr';
export type Arrival = 'periodic' | 'bursty';

export interface Ev {
  t: number;
  kind: 'rpc' | 'bulk';
  bytes: number;
}

export interface FlowOpts {
  seed?: number;
  simMs?: number;
  rpcQps?: number;
  rpcBytes?: number;
  bulkBytes?: number;
  periodMs?: number;
  bursty?: boolean;
}

export interface SimConfig {
  policy: Policy;
  wrr?: [number, number];
  /** 大包分片大小(KB)；null / 0 表示不分片 */
  chunkKb?: number | null;
  /** 只记录该时刻之前的传输/队列，用于动画窗口 */
  captureUntilMs?: number;
}

export interface Transmission {
  t0: number;
  t1: number;
  kind: 'rpc' | 'bulk';
  bytes: number;
  arrival: number;
}

export interface QueueSample {
  t: number;
  rpc: number;
  bulk: number;
}

export interface SimMetrics {
  p50: number;
  p99: number;
  p999: number;
  max: number;
  nRpc: number;
  bulkGbps: number;
  makespan: number;
  util: number;
}

export interface SimResult extends SimMetrics {
  transmissions: Transmission[];
  queueSamples: QueueSample[];
  latencies: number[];
  rpcPoints: { t: number; lat: number }[];
}

// ---------------------------------------------------------------- 基础工具

/** mulberry32：小而快的可播种 PRNG，返回 [0,1) */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 指数分布采样，均值 = mean */
function expo(rng: () => number, mean: number): number {
  return -Math.log(1 - rng()) * mean;
}

/** 在链路上发送 nbytes 所需时间(ms) */
export function txMs(nbytes: number, linkGbps = LINK_GBPS): number {
  return ((nbytes * 8) / (linkGbps * 1e9)) * 1e3;
}

/** 同 numpy.percentile 的线性插值实现 */
export function percentile(sortedAsc: number[], q: number): number {
  const n = sortedAsc.length;
  if (n === 0) return 0;
  if (n === 1) return sortedAsc[0];
  const pos = (n - 1) * (q / 100);
  const lo = Math.floor(pos);
  const frac = pos - lo;
  const hi = Math.min(lo + 1, n - 1);
  return sortedAsc[lo] + frac * (sortedAsc[hi] - sortedAsc[lo]);
}

/**
 * 由集合通信周期反推链路利用率。
 * util = (RPC 带宽 + 大包带宽) / 链路带宽
 */
export function utilFor(periodMs: number, opts: { rpcQps?: number; rpcBytes?: number; bulkBytes?: number } = {}): number {
  const rpcQps = opts.rpcQps ?? DEFAULT_RPC_QPS;
  const rpcBytes = opts.rpcBytes ?? DEFAULT_RPC_BYTES;
  const bulkBytes = opts.bulkBytes ?? DEFAULT_BULK_MB * 1e6;
  const rpcGbps = (rpcQps * rpcBytes * 8) / 1e9;
  const bulkGbps = ((bulkBytes / (periodMs / 1e3)) * 8) / 1e9;
  return (rpcGbps + bulkGbps) / LINK_GBPS;
}

/** 已知目标利用率反解集合通信周期(ms) */
export function periodForUtil(util: number, opts: { rpcQps?: number; rpcBytes?: number; bulkBytes?: number } = {}): number {
  const rpcQps = opts.rpcQps ?? DEFAULT_RPC_QPS;
  const rpcBytes = opts.rpcBytes ?? DEFAULT_RPC_BYTES;
  const bulkBytes = opts.bulkBytes ?? DEFAULT_BULK_MB * 1e6;
  const rpcGbps = (rpcQps * rpcBytes * 8) / 1e9;
  const bulkGbps = util * LINK_GBPS - rpcGbps;
  if (bulkGbps <= 0) return Infinity;
  return ((bulkBytes * 8) / (bulkGbps * 1e9)) * 1e3;
}

// ---------------------------------------------------------------- 流量生成

/**
 * 生成两类流量的到达事件：(时间, 类型, 字节)。
 * bursty=false：集合通信严格周期性到达（纯训练集群）。
 * bursty=true ：集合通信泊松到达，平均速率相同但有突发（弹性调度混部）。
 * 两者平均带宽完全一样，差别只在突发性 —— 而这恰恰是尾延迟的决定因素。
 */
export function genEvents(opts: FlowOpts = {}): Ev[] {
  const {
    seed = 0,
    simMs = SIM_MS,
    rpcQps = DEFAULT_RPC_QPS,
    rpcBytes = DEFAULT_RPC_BYTES,
    bulkBytes = DEFAULT_BULK_MB * 1e6,
    periodMs = DEFAULT_PERIOD_MS,
    bursty = false,
  } = opts;

  const rng = mulberry32(seed + 1);
  const ev: Ev[] = [];

  // RPC：泊松到达
  let t = 0;
  while (true) {
    t += expo(rng, 1000 / rpcQps);
    if (t < simMs) ev.push({ t, kind: 'rpc', bytes: rpcBytes });
    else break;
  }

  // 集合通信
  t = 0;
  while (t < simMs) {
    ev.push({ t, kind: 'bulk', bytes: bulkBytes });
    t += bursty ? expo(rng, periodMs) : periodMs;
  }

  ev.sort((a, b) => a.t - b.t || (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : 0));
  return ev;
}

// ---------------------------------------------------------------- 调度模拟

interface QItem {
  t: number;
  bytes: number;
}

export function simulate(events: Ev[], cfg: SimConfig, linkGbps = LINK_GBPS): SimResult {
  const policy = cfg.policy;
  const wrr = cfg.wrr ?? [1, 1];
  const chunkKb = cfg.chunkKb ?? null;
  const captureUntil = cfg.captureUntilMs ?? 0;

  const rpcQ: QItem[] = [];
  const bulkQ: QItem[] = [];
  let rpcH = 0;
  let bulkH = 0;

  const transmissions: Transmission[] = [];
  const queueSamples: QueueSample[] = [];
  const latencies: number[] = [];
  const rpcPoints: { t: number; lat: number }[] = [];

  const n = events.length;
  let i = 0;
  let now = 0;
  let bulkDone = 0;
  let rpcDone = 0;
  let wrrState = 0;

  const pushQueueSample = (t: number) => {
    queueSamples.push({ t, rpc: rpcQ.length - rpcH, bulk: bulkQ.length - bulkH });
  };

  const compact = () => {
    if (rpcH > 8192) {
      rpcQ.splice(0, rpcH);
      rpcH = 0;
    }
    if (bulkH > 8192) {
      bulkQ.splice(0, bulkH);
      bulkH = 0;
    }
  };

  while (true) {
    // 入队所有已到达的事件
    while (i < n && events[i].t <= now) {
      const e = events[i];
      if (e.kind === 'rpc') {
        rpcQ.push({ t: e.t, bytes: e.bytes });
      } else if (chunkKb) {
        const csz = Math.trunc(chunkKb * 1024);
        for (let off = 0; off < e.bytes; off += csz) {
          bulkQ.push({ t: e.t, bytes: Math.min(csz, e.bytes - off) });
        }
      } else {
        bulkQ.push({ t: e.t, bytes: e.bytes });
      }
      i++;
    }

    const rpcEmpty = rpcH >= rpcQ.length;
    const bulkEmpty = bulkH >= bulkQ.length;

    if (rpcEmpty && bulkEmpty) {
      if (i >= n) break;
      now = events[i].t;
      continue;
    }

    // ---- 选择下一个要发的包 ----
    let pick: 'rpc' | 'bulk';
    if (policy === 'fifo') {
      if (!rpcEmpty && !bulkEmpty) pick = rpcQ[rpcH].t <= bulkQ[bulkH].t ? 'rpc' : 'bulk';
      else pick = rpcEmpty ? 'bulk' : 'rpc';
    } else if (policy === 'priority') {
      pick = rpcEmpty ? 'bulk' : 'rpc';
    } else {
      // wrr
      const [wr, wb] = wrr;
      if (!rpcEmpty && !bulkEmpty) {
        pick = wrrState % (wr + wb) < wr ? 'rpc' : 'bulk';
        wrrState++;
      } else {
        pick = rpcEmpty ? 'bulk' : 'rpc';
      }
    }

    const q = pick === 'rpc' ? rpcQ : bulkQ;
    const h = pick === 'rpc' ? rpcH : bulkH;
    const item = q[h];
    if (pick === 'rpc') rpcH++;
    else bulkH++;

    const start = Math.max(now, item.t);
    const dur = txMs(item.bytes, linkGbps);
    now = start + dur;

    if (start <= captureUntil) {
      transmissions.push({ t0: start, t1: now, kind: pick, bytes: item.bytes, arrival: item.t });
      pushQueueSample(now);
    }

    if (pick === 'rpc') {
      const lat = now - item.t;
      latencies.push(lat);
      rpcDone += item.bytes;
      if (item.t <= captureUntil) rpcPoints.push({ t: item.t, lat });
    } else {
      bulkDone += item.bytes;
    }

    compact();
  }

  latencies.sort((a, b) => a - b);
  const makespan = now;
  return {
    transmissions,
    queueSamples,
    latencies,
    rpcPoints,
    p50: percentile(latencies, 50),
    p99: percentile(latencies, 99),
    p999: percentile(latencies, 99.9),
    max: latencies.length ? latencies[latencies.length - 1] : 0,
    nRpc: latencies.length,
    bulkGbps: (bulkDone * 8) / (makespan / 1e3) / 1e9,
    makespan,
    util: ((bulkDone + rpcDone) * 8) / (makespan / 1e3) / 1e9 / linkGbps,
  };
}
