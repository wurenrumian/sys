<script setup lang="ts">
/**
 * TrafficConflictDemo —— 03_network/demo1 的浏览器内实时模拟。
 *
 * 读者可以亲手调：调度策略、是否分片、集合通信是周期性还是突发、
 * 链路利用率，然后看着链路被大包占住、RPC 在队列里排队，
 * 并实时读出 P50/P99/P999。结论不再需要"相信我"。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  LINK_GBPS,
  DEFAULT_RPC_BYTES,
  DEFAULT_BULK_MB,
  genEvents,
  simulate,
  txMs,
  utilFor,
  periodForUtil,
  type SimResult,
  type SimMetrics,
  type Transmission,
} from '../lib/queueSim'

interface Strategy {
  id: string
  label: string
  policy: 'fifo' | 'priority' | 'wrr'
  wrr?: [number, number]
  chunkKb?: number
}

const STRATEGIES: Strategy[] = [
  { id: 'fifo', label: 'FIFO 单队列', policy: 'fifo' },
  { id: 'priority', label: '优先级(RPC优先)', policy: 'priority' },
  { id: 'wrr1', label: '加权轮转 1:1', policy: 'wrr', wrr: [1, 1] },
  { id: 'wrr4', label: '加权轮转 4:1', policy: 'wrr', wrr: [4, 1] },
  { id: 'prio64', label: '优先级+分片64KB', policy: 'priority', chunkKb: 64 },
  { id: 'prio4', label: '优先级+分片4KB', policy: 'priority', chunkKb: 4 },
]

const isClient = typeof window !== 'undefined' && typeof requestAnimationFrame !== 'undefined'

const BULK_BYTES = DEFAULT_BULK_MB * 1e6
const bulkTxUs = txMs(BULK_BYTES) * 1e3
const rpcTxUs = txMs(DEFAULT_RPC_BYTES) * 1e3

// ---------------------------------------------------------------- 状态
const strategyId = ref('fifo')
const bursty = ref(false)
const util = ref(0.16)
const seed = ref(0)
const windowMs = ref(2)
const WINDOWS = [0.3, 1, 2, 5, 10]

const active = computed(() => STRATEGIES.find((s) => s.id === strategyId.value)!)

const periodMs = computed(() => periodForUtil(util.value))
const events = computed(() =>
  genEvents({ seed: seed.value, periodMs: periodMs.value, bursty: bursty.value }),
)

const current = ref<SimResult | null>(null)
const table = ref<{ id: string; label: string; m: SimMetrics }[]>([])
const busy = ref(false)

let timer: ReturnType<typeof setTimeout> | undefined

function configOf(s: Strategy, capture: number) {
  return { policy: s.policy, wrr: s.wrr, chunkKb: s.chunkKb ?? null, captureUntilMs: capture }
}

function runAll() {
  busy.value = true
  clearTimeout(timer)
  timer = setTimeout(() => {
    const ev = events.value
    current.value = simulate(ev, configOf(active.value, windowMs.value), LINK_GBPS)
    table.value = STRATEGIES.map((s) => ({
      id: s.id,
      label: s.label,
      m: simulate(ev, configOf(s, 0), LINK_GBPS),
    }))
    busy.value = false
    restart()
  }, 40)
}

function runCurrent() {
  clearTimeout(timer)
  timer = setTimeout(() => {
    current.value = simulate(events.value, configOf(active.value, windowMs.value), LINK_GBPS)
    restart()
  }, 30)
}

// 只在浏览器端启动：SSR / prerender 阶段没有 rAF 与 canvas
watch([strategyId, bursty, util, seed], () => {
  if (isClient) runAll()
})
watch(windowMs, () => {
  if (isClient) runCurrent()
})

// ---------------------------------------------------------------- 利用率扫描
interface ScanRow {
  util: number
  periodic: number
  burstyFifo: number
  burstyChunk: number
}
const scan = ref<ScanRow[]>([])
const scanning = ref(false)
const SCAN_UTILS = [0.16, 0.32, 0.64, 0.8, 0.89, 0.93]

function runScan() {
  scanning.value = true
  setTimeout(() => {
    scan.value = SCAN_UTILS.map((u) => {
      const p = periodForUtil(u)
      const evp = genEvents({ seed: seed.value, periodMs: p, bursty: false })
      const evb = genEvents({ seed: seed.value, periodMs: p, bursty: true })
      return {
        util: u,
        periodic: simulate(evp, { policy: 'fifo', captureUntilMs: 0 }, LINK_GBPS).p999 * 1e3,
        burstyFifo: simulate(evb, { policy: 'fifo', captureUntilMs: 0 }, LINK_GBPS).p999 * 1e3,
        burstyChunk: simulate(evb, { policy: 'priority', chunkKb: 4, captureUntilMs: 0 }).p999 * 1e3,
      }
    })
    scanning.value = false
  }, 30)
}
watch(seed, runScan)

// ---------------------------------------------------------------- 回放动画
const canvas = ref<HTMLCanvasElement | null>(null)
const wrap = ref<HTMLElement | null>(null)
const playing = ref(true)
const progress = ref(0)
const speed = ref(1)
const SPEEDS = [0.5, 1, 2, 4]

const live = ref({
  timeUs: 0,
  link: '空闲',
  linkKind: 'idle' as 'idle' | 'rpc' | 'bulk',
  rpcQ: 0,
  bulkQ: 0,
  lastRpcLatUs: 0,
})

let raf = 0
let prevTs = 0
let dpr = 1

function msPerSecond() {
  return 0.55 * speed.value
}

function frame(ts: number) {
  if (!playing.value) return
  if (!prevTs) prevTs = ts
  const dt = (ts - prevTs) / 1000
  prevTs = ts
  progress.value += dt * msPerSecond()
  if (progress.value >= windowMs.value) {
    progress.value = windowMs.value
    playing.value = false
    draw()
    return
  }
  draw()
  raf = requestAnimationFrame(frame)
}

function start() {
  if (!isClient || playing.value) return
  playing.value = true
  prevTs = 0
  cancelAnimationFrame(raf)
  raf = requestAnimationFrame(frame)
}
function pause() {
  playing.value = false
  if (isClient) cancelAnimationFrame(raf)
}
function restart() {
  if (!isClient) {
    progress.value = 0
    return
  }
  cancelAnimationFrame(raf)
  progress.value = 0
  playing.value = true
  prevTs = 0
  raf = requestAnimationFrame(frame)
}

function xOf(t: number, W: number, pad: number) {
  return pad + (t / windowMs.value) * (W - pad * 2)
}

function draw() {
  const c = canvas.value
  const r = current.value
  if (!c || !r) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  const W = c.width / dpr
  const H = c.height / dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)

  const pad = 66
  const laneA = { y: 20, h: 30 }
  const laneB = { y: 74, h: 40 }

  // 背景与网格
  ctx.fillStyle = 'rgba(127,127,127,0.06)'
  ctx.fillRect(pad, laneA.y, W - pad * 2, laneA.h)
  ctx.fillRect(pad, laneB.y, W - pad * 2, laneB.h)

  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
  ctx.fillStyle = 'rgba(127,127,127,0.9)'
  ctx.textAlign = 'center'
  const ticks = 5
  for (let i = 0; i <= ticks; i++) {
    const t = (windowMs.value / ticks) * i
    const x = xOf(t, W, pad)
    ctx.strokeStyle = 'rgba(127,127,127,0.18)'
    ctx.beginPath()
    ctx.moveTo(x, laneA.y)
    ctx.lineTo(x, laneB.y + laneB.h)
    ctx.stroke()
    ctx.fillText(`${(t * 1e3).toFixed(t * 1e3 < 10 ? 1 : 0)}µs`, x, 12)
  }

  // 链路占用（Lane A）
  const t = progress.value
  const chunked = !!active.value.chunkKb
  for (const tr of r.transmissions) {
    if (tr.t0 > t) break
    if (tr.kind === 'rpc' && chunked) {
      // 分片模式下 RPC 依然逐个存在，但视觉上太细，保留最小宽度
    }
    const x0 = xOf(tr.t0, W, pad)
    const x1 = xOf(Math.min(tr.t1, t), W, pad)
    const w = Math.max(chunked && tr.kind === 'bulk' ? 0.6 : 1.4, x1 - x0)
    ctx.fillStyle = tr.kind === 'bulk' ? '#3b82f6' : '#f59e0b'
    ctx.globalAlpha = tr.t1 <= t ? 0.95 : 0.55
    ctx.fillRect(x0, laneA.y + 5, w, laneA.h - 10)
  }
  ctx.globalAlpha = 1

  // Lane B：每个 RPC 的延迟（x = 到达时刻，竖线高度 = 等待时间，对数刻度）
  // 这是让「队头阻塞」真正可见的一层：撞上大包的 RPC 会从 320µs 高度慢慢滑落。
  const latBot = Math.log10(0.02)
  const latTop = Math.log10(bulkTxUs)
  const latY = (latUs: number) => {
    const f = (Math.log10(Math.max(latUs, 0.02)) - latBot) / (latTop - latBot)
    return laneB.y + laneB.h - Math.min(1, Math.max(0, f)) * (laneB.h - 4)
  }

  // 不可抢占窗口参考线
  const yPre = latY(preemptUs.value)
  ctx.strokeStyle = 'rgba(127,127,127,0.55)'
  ctx.setLineDash([4, 3])
  ctx.beginPath()
  ctx.moveTo(pad, yPre)
  ctx.lineTo(W - pad, yPre)
  ctx.stroke()
  ctx.setLineDash([])
  ctx.fillStyle = 'rgba(127,127,127,0.9)'
  ctx.textAlign = 'left'
  ctx.fillText(`不可抢占 ${preemptUs.value >= 1 ? preemptUs.value.toFixed(0) : preemptUs.value.toFixed(2)}µs`, pad + 4, yPre - 3)

  for (const p of r.rpcPoints) {
    if (p.t > t) continue
    const x = xOf(p.t, W, pad)
    const y = latY(p.lat * 1e3)
    ctx.strokeStyle = 'rgba(245,158,11,0.55)'
    ctx.beginPath()
    ctx.moveTo(x, laneB.y + laneB.h)
    ctx.lineTo(x, y)
    ctx.stroke()
    ctx.fillStyle = '#f59e0b'
    ctx.beginPath()
    ctx.arc(x, y, 1.6, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.fillStyle = 'rgba(127,127,127,0.95)'
  ctx.textAlign = 'right'
  ctx.fillText('链路占用', pad - 8, laneA.y + laneA.h / 2 + 3)
  ctx.fillText('RPC 延迟', pad - 8, laneB.y + 10)

  // 播放头
  const px = xOf(t, W, pad)
  ctx.strokeStyle = '#ef4444'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(px, laneA.y - 4)
  ctx.lineTo(px, laneB.y + laneB.h + 4)
  ctx.stroke()
  ctx.lineWidth = 1

  // 实时读数
  let cur: Transmission | null = null
  let lastLat = 0
  for (const tr of r.transmissions) {
    if (tr.t0 > t) break
    if (tr.t0 <= t && tr.t1 >= t) cur = tr
    if (tr.kind === 'rpc') lastLat = (tr.t1 - tr.arrival) * 1e3
  }
  let qr = 0
  let qb = 0
  for (const s of r.queueSamples) {
    if (s.t > t) break
    qr = s.rpc
    qb = s.bulk
  }
  live.value = {
    timeUs: t * 1e3,
    link: cur
      ? cur.kind === 'bulk'
        ? active.value.chunkKb
          ? `大包分片 ${active.value.chunkKb}KB`
          : `集合通信大包 ${DEFAULT_BULK_MB}MB`
        : 'RPC 小包 512B'
      : '空闲',
    linkKind: (cur ? cur.kind : 'idle') as 'idle' | 'rpc' | 'bulk',
    rpcQ: qr,
    bulkQ: qb,
    lastRpcLatUs: lastLat,
  }
}

function resize() {
  const c = canvas.value
  const w = wrap.value
  if (!c || !w) return
  dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = w.clientWidth
  const height = 130
  c.width = Math.round(width * dpr)
  c.height = Math.round(height * dpr)
  c.style.width = width + 'px'
  c.style.height = height + 'px'
  draw()
}

let ro: ResizeObserver | null = null
onMounted(() => {
  runAll()
  resize()
  runScan()
  if (typeof ResizeObserver !== 'undefined' && wrap.value) {
    ro = new ResizeObserver(resize)
    ro.observe(wrap.value)
  }
})
onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  clearTimeout(timer)
  ro?.disconnect()
})

// ---------------------------------------------------------------- 扫描图（SVG）
const CHART = { w: 660, h: 280, padL: 54, padR: 16, padT: 18, padB: 40 }
const scanMax = computed(() => {
  let m = 1
  for (const r of scan.value) m = Math.max(m, r.periodic, r.burstyFifo, r.burstyChunk)
  return m
})
const scanYMin = 0.2
const logMin = computed(() => Math.log10(scanYMin))
const logMax = computed(() => Math.log10(scanMax.value * 1.25))

const X_MIN = 0.12
const X_MAX = 0.95
function sx(u: number) {
  const c = Math.min(X_MAX, Math.max(X_MIN, u))
  return CHART.padL + ((c - X_MIN) / (X_MAX - X_MIN)) * (CHART.w - CHART.padL - CHART.padR)
}
function sy(v: number) {
  const lv = Math.max(logMin.value, Math.log10(Math.max(v, scanYMin)))
  const f = (lv - logMin.value) / (logMax.value - logMin.value)
  return CHART.h - CHART.padB - f * (CHART.h - CHART.padT - CHART.padB)
}
function path(key: 'periodic' | 'burstyFifo' | 'burstyChunk') {
  return scan.value.map((r, i) => `${i ? 'L' : 'M'}${sx(r.util).toFixed(1)},${sy(r[key]).toFixed(1)}`).join(' ')
}
const yTicks = computed(() => {
  const out: number[] = []
  for (let e = -1; e <= Math.ceil(logMax.value); e++) {
    const v = Math.pow(10, e)
    if (v >= scanYMin && v <= scanMax.value * 1.25) out.push(v)
  }
  return out
})
function fmtUs(v: number) {
  if (v >= 1000) return (v / 1000).toFixed(1) + 'ms'
  if (v >= 1) return v.toFixed(0) + 'µs'
  return v.toFixed(2) + 'µs'
}

// ---------------------------------------------------------------- 展示工具
function fmtLat(ms: number) {
  const us = ms * 1e3
  if (us >= 1000) return (us / 1000).toFixed(1) + ' ms'
  return us.toFixed(us < 10 ? 1 : 0) + ' µs'
}
const currentUtil = computed(() => utilFor(periodMs.value) * 100)

// 尾延迟的下界 = 最大不可抢占单元的传输时间
const preemptUs = computed(() =>
  active.value.chunkKb ? txMs(active.value.chunkKb * 1024) * 1e3 : bulkTxUs,
)
const preemptLabel = computed(() =>
  preemptUs.value >= 1 ? preemptUs.value.toFixed(1) + ' µs' : preemptUs.value.toFixed(3) + ' µs',
)
</script>

<template>
  <div class="tfc card">
    <div class="tfc__head">
      <div>
        <div class="tfc__eyebrow">03 / demo1 · 实时模拟</div>
        <h3 class="tfc__title">队头阻塞：一根网线上同时跑「搬家卡车」和「救护车」</h3>
      </div>
      <div class="tfc__badge" :class="{ 'is-busy': busy }">
        {{ busy ? '计算中…' : '模型与脚本一致' }}
      </div>
    </div>

    <!-- 参数区 -->
    <div class="tfc__controls">
      <div class="tfc__ctrl">
        <label>调度策略</label>
        <div class="tfc__seg tfc__seg--wrap">
          <button
            v-for="s in STRATEGIES"
            :key="s.id"
            :class="{ 'is-on': strategyId === s.id }"
            @click="strategyId = s.id"
          >
            {{ s.label }}
          </button>
        </div>
      </div>

      <div class="tfc__ctrl">
        <label>集合通信到达</label>
        <div class="tfc__seg">
          <button :class="{ 'is-on': !bursty }" @click="bursty = false">周期性（纯训练）</button>
          <button :class="{ 'is-on': bursty }" @click="bursty = true">突发（弹性混部）</button>
        </div>
      </div>

      <div class="tfc__ctrl">
        <label>
          链路利用率 <b>{{ currentUtil.toFixed(0) }}%</b>
          <span class="tfc__hint">（集合通信每 {{ periodMs.toFixed(2) }}ms 一个 4MB 块）</span>
        </label>
        <input v-model.number="util" type="range" min="0.05" max="0.95" step="0.01" />
      </div>

      <div class="tfc__ctrl tfc__ctrl--row">
        <div>
          <label>动画取景窗口</label>
          <div class="tfc__seg">
            <button
              v-for="w in WINDOWS"
              :key="w"
              :class="{ 'is-on': windowMs === w }"
              @click="windowMs = w"
            >
              {{ w < 1 ? w * 1000 + 'µs' : w + 'ms' }}
            </button>
          </div>
        </div>
        <button class="tfc__newseed" @click="seed++">重新生成流量 ↻</button>
      </div>
    </div>

    <!-- 实时指标 -->
    <div v-if="current" class="tfc__metrics">
      <div class="tfc__metric">
        <span>RPC P50</span><b>{{ fmtLat(current.p50) }}</b>
      </div>
      <div class="tfc__metric">
        <span>RPC P99</span><b>{{ fmtLat(current.p99) }}</b>
      </div>
      <div class="tfc__metric tfc__metric--hot">
        <span>RPC P999</span><b>{{ fmtLat(current.p999) }}</b>
      </div>
      <div class="tfc__metric">
        <span>RPC 最大</span><b>{{ fmtLat(current.max) }}</b>
      </div>
      <div class="tfc__metric">
        <span>训练吞吐</span><b>{{ current.bulkGbps.toFixed(1) }} Gbps</b>
      </div>
      <div class="tfc__metric">
        <span>不可抢占窗口</span><b>{{ preemptLabel }}</b>
      </div>
    </div>

    <!-- 动画 -->
    <div ref="wrap" class="tfc__scope">
      <canvas ref="canvas" />
    </div>
    <div class="tfc__playbar">
      <button class="tfc__btn" @click="playing ? pause() : start()">
        {{ playing ? '⏸ 暂停' : '▶ 播放' }}
      </button>
      <button class="tfc__btn" @click="restart">↺ 重播</button>
      <span class="tfc__seg tfc__seg--sm">
        <button
          v-for="s in SPEEDS"
          :key="s"
          :class="{ 'is-on': speed === s }"
          @click="speed = s"
        >
          {{ s }}x
        </button>
      </span>
      <span class="tfc__legend">
        <i class="sw sw--bulk" /> 集合通信大包
        <i class="sw sw--rpc" /> RPC 小包
      </span>
    </div>
    <div class="tfc__live">
      <span>t = {{ live.timeUs.toFixed(1) }}µs</span>
      <span>链路：<b :class="'k-' + live.linkKind">{{ live.link }}</b></span>
      <span>RPC 队列 <b class="k-rpc">{{ live.rpcQ }}</b></span>
      <span>大包队列 <b class="k-bulk">{{ live.bulkQ }}</b></span>
      <span>最近 RPC 延迟 <b>{{ live.lastRpcLatUs.toFixed(1) }}µs</b></span>
    </div>

    <p class="tfc__read">
      单个 4MB 大包要占住链路 <b>{{ bulkTxUs.toFixed(0) }}µs</b>，而 RPC 包自身只需
      <b>{{ rpcTxUs.toFixed(3) }}µs</b> —— 相差
      <b>{{ Math.round(bulkTxUs / rpcTxUs).toLocaleString() }} 倍</b>。所以 FIFO 下 RPC 的尾延迟
      几乎精确等于一个大包的发送时间。试着把策略切到「优先级」，你会发现
      <b>几乎没有改善</b>：包一旦开始发送就不可抢占。真正治本的是把大包切小。
    </p>

    <!-- 策略对比表 -->
    <div class="tfc__block">
      <h4>同一份流量下，各调度策略的表现</h4>
      <p class="tfc__sub">
        当前流量：{{ bursty ? '突发（弹性混部）' : '周期性（纯训练）' }}，利用率
        {{ currentUtil.toFixed(0) }}%。点任意一行可切换。
      </p>
      <div class="tfc__tablewrap">
        <table class="tfc__table">
          <thead>
            <tr>
              <th>调度策略</th>
              <th>P50</th>
              <th>P99</th>
              <th>P999</th>
              <th>最大</th>
              <th>训练吞吐</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in table"
              :key="row.id"
              :class="{ 'is-active': row.id === strategyId }"
              @click="strategyId = row.id"
            >
              <td class="tfc__tdname">{{ row.label }}</td>
              <td>{{ fmtLat(row.m.p50) }}</td>
              <td>{{ fmtLat(row.m.p99) }}</td>
              <td class="tfc__tdhot">{{ fmtLat(row.m.p999) }}</td>
              <td>{{ fmtLat(row.m.max) }}</td>
              <td>{{ row.m.bulkGbps.toFixed(1) }} Gbps</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 利用率 × 突发性 扫描 -->
    <div class="tfc__block">
      <h4>反直觉结论：决定尾延迟的是「突发性」，不是平均利用率</h4>
      <p class="tfc__sub">
        同样把利用率从 16% 推到 93%：周期性流量下 P999 纹丝不动，
        突发流量下 P999 却涨了两个数量级。
      </p>
      <div class="tfc__chart">
        <svg :viewBox="`0 0 ${CHART.w} ${CHART.h}`" preserveAspectRatio="xMidYMid meet">
          <!-- 网格与 y 轴 -->
          <g>
            <line
              v-for="v in yTicks"
              :key="'y' + v"
              :x1="CHART.padL"
              :x2="CHART.w - CHART.padR"
              :y1="sy(v)"
              :y2="sy(v)"
              class="grid"
            />
            <text
              v-for="v in yTicks"
              :key="'yl' + v"
              :x="CHART.padL - 6"
              :y="sy(v) + 3"
              text-anchor="end"
              class="axis"
            >
              {{ fmtUs(v) }}
            </text>
          </g>
          <!-- x 轴 -->
          <g>
            <text
              v-for="r in scan"
              :key="'x' + r.util"
              :x="sx(r.util)"
              :y="CHART.h - CHART.padB + 16"
              text-anchor="middle"
              class="axis"
            >
              {{ (r.util * 100).toFixed(0) }}%
            </text>
            <text
              :x="(CHART.padL + CHART.w - CHART.padR) / 2"
              :y="CHART.h - 6"
              text-anchor="middle"
              class="axis axis--title"
            >
              链路利用率
            </text>
          </g>
          <text
            :x="14"
            :y="(CHART.padT + CHART.h - CHART.padB) / 2"
            text-anchor="middle"
            class="axis axis--title"
            :transform="`rotate(-90 14 ${(CHART.padT + CHART.h - CHART.padB) / 2})`"
          >
            RPC P999（对数轴）
          </text>
          <!-- 当前利用率标线 -->
          <line
            :x1="sx(util)"
            :x2="sx(util)"
            :y1="CHART.padT"
            :y2="CHART.h - CHART.padB"
            class="marker"
          />
          <!-- 三条曲线 -->
          <path :d="path('burstyFifo')" class="line line--danger" />
          <path :d="path('burstyChunk')" class="line line--ok" />
          <path :d="path('periodic')" class="line line--brand" />
          <g v-for="r in scan" :key="'p' + r.util">
            <circle :cx="sx(r.util)" :cy="sy(r.burstyFifo)" r="3" class="dot dot--danger">
              <title>突发 FIFO：{{ fmtUs(r.burstyFifo) }}</title>
            </circle>
            <circle :cx="sx(r.util)" :cy="sy(r.periodic)" r="3" class="dot dot--brand">
              <title>周期性 FIFO：{{ fmtUs(r.periodic) }}</title>
            </circle>
            <circle :cx="sx(r.util)" :cy="sy(r.burstyChunk)" r="3" class="dot dot--ok">
              <title>突发 + 分片4KB：{{ fmtUs(r.burstyChunk) }}</title>
            </circle>
          </g>
        </svg>
        <div v-if="scanning" class="tfc__scanning">扫描中…</div>
      </div>
      <div class="tfc__legend tfc__legend--block">
        <span><i class="sw sw--brand" /> 周期性 FIFO</span>
        <span><i class="sw sw--danger" /> 突发 FIFO</span>
        <span><i class="sw sw--ok" /> 突发 + 分片4KB</span>
      </div>
    </div>

    <p class="tfc__foot">
      模型与 <code>03_network/demo1_traffic_conflict.py</code> 一致（store-and-forward、不可抢占）；
      随机数发生器不同（脚本用 numpy，这里用 mulberry32），数值有统计波动。
    </p>
  </div>
</template>

<style scoped>
.tfc {
  margin: 20px 0;
  padding: 18px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
}
.tfc__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 14px;
}
.tfc__eyebrow {
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vp-c-brand-1);
  font-weight: 700;
}
.tfc__title {
  margin: 4px 0 0;
  font-size: 17px;
  line-height: 1.5;
}
.tfc__badge {
  flex: none;
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 999px;
  color: var(--vp-c-green-1, #10b981);
  background: color-mix(in srgb, var(--vp-c-green-1, #10b981) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--vp-c-green-1, #10b981) 35%, transparent);
}
.tfc__badge.is-busy {
  color: var(--vp-c-warning-1, #f59e0b);
  background: color-mix(in srgb, var(--vp-c-warning-1, #f59e0b) 14%, transparent);
  border-color: color-mix(in srgb, var(--vp-c-warning-1, #f59e0b) 35%, transparent);
}
.tfc__controls {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
}
.tfc__ctrl label {
  display: block;
  font-size: 12.5px;
  color: var(--vp-c-text-2);
  margin-bottom: 6px;
}
.tfc__ctrl label b {
  color: var(--vp-c-text-1);
}
.tfc__hint {
  color: var(--vp-c-text-3);
  font-size: 11.5px;
}
.tfc__ctrl--row {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
}
.tfc__seg {
  display: inline-flex;
  gap: 4px;
  flex-wrap: wrap;
}
.tfc__seg button,
.tfc__btn,
.tfc__newseed {
  font: inherit;
  font-size: 12.5px;
  padding: 4px 10px;
  border-radius: 7px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.15s;
}
.tfc__seg button:hover,
.tfc__btn:hover,
.tfc__newseed:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-text-1);
}
.tfc__seg button.is-on {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #fff;
}
.tfc__seg--sm button {
  padding: 3px 8px;
  font-size: 12px;
}
input[type='range'] {
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}
.tfc__metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 8px;
  margin: 14px 0;
}
.tfc__metric {
  padding: 8px 10px;
  border-radius: 9px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}
.tfc__metric span {
  display: block;
  font-size: 11px;
  color: var(--vp-c-text-3);
}
.tfc__metric b {
  font-size: 15.5px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.tfc__metric--hot {
  border-color: color-mix(in srgb, var(--vp-c-danger-1, #ef4444) 45%, transparent);
}
.tfc__metric--hot b {
  color: var(--vp-c-danger-1, #ef4444);
}
.tfc__scope {
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
  overflow: hidden;
}
.tfc__scope canvas {
  display: block;
}
.tfc__playbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0 4px;
  flex-wrap: wrap;
}
.tfc__legend {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.tfc__legend--block {
  margin: 8px 0 0;
  flex-wrap: wrap;
}
.tfc__legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-right: 14px;
}
.sw {
  width: 11px;
  height: 11px;
  border-radius: 3px;
  display: inline-block;
}
.sw--bulk,
.sw--brand {
  background: #3b82f6;
}
.sw--rpc {
  background: #f59e0b;
}
.sw--danger {
  background: #ef4444;
}
.sw--ok {
  background: #10b981;
}
.tfc__live {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  padding: 8px 10px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  border-radius: 8px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
}
.tfc__live .k-bulk {
  color: #3b82f6;
}
.tfc__live .k-rpc {
  color: #f59e0b;
}
.tfc__live .k-idle {
  color: var(--vp-c-text-3);
}
.tfc__read {
  font-size: 13.5px;
  line-height: 1.75;
  color: var(--vp-c-text-2);
  margin: 14px 0 0;
}
.tfc__read b {
  color: var(--vp-c-text-1);
}
.tfc__block {
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px solid var(--vp-c-divider);
}
.tfc__block h4 {
  margin: 0 0 4px;
  font-size: 15px;
}
.tfc__sub {
  margin: 0 0 12px;
  font-size: 12.5px;
  color: var(--vp-c-text-3);
}
.tfc__tablewrap {
  overflow-x: auto;
}
.tfc__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.tfc__table th,
.tfc__table td {
  padding: 7px 10px;
  text-align: right;
  border-bottom: 1px solid var(--vp-c-divider);
  white-space: nowrap;
}
.tfc__table th:first-child,
.tfc__table td:first-child {
  text-align: left;
}
.tfc__table th {
  font-weight: 600;
  color: var(--vp-c-text-3);
  font-size: 12px;
}
.tfc__table tbody tr {
  cursor: pointer;
  transition: background 0.12s;
}
.tfc__table tbody tr:hover {
  background: var(--vp-c-bg-soft);
}
.tfc__table tbody tr.is-active {
  background: color-mix(in srgb, var(--vp-c-brand-1) 12%, transparent);
}
.tfc__tdname {
  font-weight: 600;
}
.tfc__tdhot {
  color: var(--vp-c-danger-1, #ef4444);
}
.tfc__chart {
  position: relative;
  width: 100%;
}
.tfc__chart svg {
  width: 100%;
  height: auto;
  display: block;
}
.tfc__chart .grid {
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}
.tfc__chart .axis {
  fill: var(--vp-c-text-3);
  font-size: 11px;
  font-family: var(--vp-font-family-mono);
}
.tfc__chart .axis--title {
  font-family: inherit;
  font-size: 11.5px;
}
.tfc__chart .marker {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.tfc__chart .line {
  fill: none;
  stroke-width: 2;
}
.tfc__chart .line--brand {
  stroke: #3b82f6;
}
.tfc__chart .line--danger {
  stroke: #ef4444;
}
.tfc__chart .line--ok {
  stroke: #10b981;
}
.tfc__chart .dot {
  stroke: var(--vp-c-bg);
  stroke-width: 1.5;
}
.tfc__chart .dot--brand {
  fill: #3b82f6;
}
.tfc__chart .dot--danger {
  fill: #ef4444;
}
.tfc__chart .dot--ok {
  fill: #10b981;
}
.tfc__scanning {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: 12px;
  color: var(--vp-c-text-3);
  background: color-mix(in srgb, var(--vp-c-bg) 60%, transparent);
}
.tfc__foot {
  margin: 16px 0 0;
  font-size: 11.5px;
  color: var(--vp-c-text-3);
}
</style>
