<script setup lang="ts">
/**
 * DynamicBatchDemo —— 02_compile/demo4 的浏览器版。
 * 攒批要等，等就是延迟：固定批 / 固定超时 / 自适应在波动+尖峰流量下的取舍。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { BASE_QPS, SIM_MS, SPIKE_AT, genArrivals, simulateBatch, type BatchConfig } from '../lib/batch'
import { fmtMs } from '../lib/format'

const isClient = typeof window !== 'undefined'

interface Preset {
  key: string
  name: string
  cfg: BatchConfig
}
const PRESETS: Preset[] = [
  { key: 'bs16', name: '固定批 bs=16', cfg: { policy: 'fixed_bs', fixedBs: 16 } },
  { key: 'bs64', name: '固定批 bs=64', cfg: { policy: 'fixed_bs', fixedBs: 64 } },
  { key: 'bs256', name: '固定批 bs=256', cfg: { policy: 'fixed_bs', fixedBs: 256 } },
  { key: 'to5', name: '固定超时 5ms', cfg: { policy: 'fixed_timeout', timeoutMs: 5 } },
  { key: 'to20', name: '固定超时 20ms', cfg: { policy: 'fixed_timeout', timeoutMs: 20 } },
  { key: 'adaptive', name: '自适应', cfg: { policy: 'adaptive' } },
]

const qps = ref(BASE_QPS)
const selected = ref('adaptive')
const busy = ref(false)
const table = ref<Array<{ key: string; name: string; p50: number; p99: number; p999: number; max: number; avgBs: number; util: number; latencies: Float64Array }>>([])
const arrivals = ref<Float64Array>(new Float64Array())
const batches = ref<Array<{ t0: number; t1: number; bs: number }>>([])

let timer: ReturnType<typeof setTimeout> | undefined
function recompute() {
  busy.value = true
  clearTimeout(timer)
  timer = setTimeout(() => {
    const arr = genArrivals(0, qps.value)
    arrivals.value = arr
    table.value = PRESETS.map((p) => {
      const r = simulateBatch(arr, p.cfg)
      return { key: p.key, name: p.name, p50: r.p50, p99: r.p99, p999: r.p999, max: r.max, avgBs: r.avgBs, util: r.gpuUtil, latencies: r.latencies }
    })
    const sel = PRESETS.find((p) => p.key === selected.value)!
    batches.value = simulateBatch(arr, sel.cfg).batches
    busy.value = false
    restart()
  }, 50)
}
if (isClient) watch([qps, selected], recompute)

const current = computed(() => table.value.find((t) => t.key === selected.value))

// 尖峰/低谷分段
const seg = computed(() => {
  const arr = arrivals.value
  const spike = (i: number) => i < arr.length && arr[i] >= SPIKE_AT[0] && arr[i] < SPIKE_AT[1]
  return table.value.map((t) => {
    const sLat: number[] = []
    const tLat: number[] = []
    for (let i = 0; i < arr.length; i++) (spike(i) ? sLat : tLat).push(t.latencies[i])
    const pct = (a: number[], p: number) => {
      if (!a.length) return 0
      const s = a.sort((x, y) => x - y)
      const pos = (s.length - 1) * (p / 100)
      const lo = Math.floor(pos)
      const hi = Math.min(lo + 1, s.length - 1)
      return s[lo] + (pos - lo) * (s[hi] - s[lo])
    }
    return { key: t.key, name: t.name, troughP99: pct(tLat, 99), spikeP99: pct(sLat, 99), troughP999: pct(tLat, 99.9), spikeP999: pct(sLat, 99.9) }
  })
})

// ---------------------------------------------------------------- 动画
const canvas = ref<HTMLCanvasElement | null>(null)
const wrap = ref<HTMLElement | null>(null)
const playing = ref(true)
const progress = ref(0)
const speed = ref(1)
const SPEEDS = [0.5, 1, 2, 4]
const live = ref({ t: 0, bs: 0, lat: 0 })

const BUCKET = 200
const NBUCKET = Math.ceil(SIM_MS / BUCKET)
const bucketBs = computed(() => {
  const sum = new Float64Array(NBUCKET)
  const cnt = new Float64Array(NBUCKET)
  for (const b of batches.value) {
    const k = Math.min(NBUCKET - 1, Math.floor(b.t0 / BUCKET))
    sum[k] += b.bs
    cnt[k]++
  }
  const out = new Float64Array(NBUCKET)
  for (let i = 0; i < NBUCKET; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : 0
  return out
})
const bucketLat = computed(() => {
  const cur = current.value
  if (!cur) return new Float64Array(NBUCKET)
  const sum = new Float64Array(NBUCKET)
  const cnt = new Float64Array(NBUCKET)
  const arr = arrivals.value
  for (let i = 0; i < arr.length; i++) {
    const k = Math.min(NBUCKET - 1, Math.floor(arr[i] / BUCKET))
    sum[k] += cur.latencies[i]
    cnt[k]++
  }
  const out = new Float64Array(NBUCKET)
  for (let i = 0; i < NBUCKET; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : 0
  return out
})
const maxBsBucket = computed(() => Math.max(1, ...bucketBs.value))
const maxLatBucket = computed(() => Math.max(1, ...bucketLat.value))

let raf = 0
let prevTs = 0
let dpr = 1
function frame(ts: number) {
  if (!playing.value) return
  if (!prevTs) prevTs = ts
  const dt = (ts - prevTs) / 1000
  prevTs = ts
  progress.value += dt * 12000 * speed.value // ms sim / s real
  if (progress.value >= SIM_MS) {
    progress.value = SIM_MS
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
  if (!isClient) return
  cancelAnimationFrame(raf)
  progress.value = 0
  playing.value = true
  prevTs = 0
  raf = requestAnimationFrame(frame)
}

function draw() {
  const c = canvas.value
  if (!c) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  const W = c.width / dpr
  const H = c.height / dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  const padL = 46
  const padB = 34
  const top = 12
  const bot = H - padB
  const xOf = (t: number) => padL + (t / SIM_MS) * (W - padL - 10)
  const tNow = progress.value

  // 尖峰区底色
  ctx.fillStyle = 'rgba(239,68,68,0.08)'
  ctx.fillRect(xOf(SPIKE_AT[0]), top, xOf(SPIKE_AT[1]) - xOf(SPIKE_AT[0]), bot - top)

  // 批大小柱
  const bw = (W - padL - 10) / NBUCKET
  for (let i = 0; i < NBUCKET; i++) {
    const t = i * BUCKET
    if (t > tNow) break
    const h = (bucketBs.value[i] / maxBsBucket.value) * (bot - top)
    ctx.fillStyle = 'rgba(59,130,246,0.75)'
    ctx.fillRect(xOf(t), bot - h, Math.max(1, bw - 1), h)
  }
  // 平均延迟折线
  ctx.strokeStyle = 'rgba(245,158,11,0.95)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  for (let i = 0; i < NBUCKET; i++) {
    const t = i * BUCKET
    if (t > tNow) break
    const y = bot - (bucketLat.value[i] / maxLatBucket.value) * (bot - top)
    i ? ctx.lineTo(xOf(t), y) : ctx.moveTo(xOf(t), y)
  }
  ctx.stroke()
  ctx.lineWidth = 1

  // 播放头
  ctx.strokeStyle = '#ef4444'
  ctx.beginPath()
  ctx.moveTo(xOf(tNow), top)
  ctx.lineTo(xOf(tNow), bot)
  ctx.stroke()

  // 时间轴
  ctx.font = '10px ui-monospace, monospace'
  ctx.fillStyle = 'rgba(127,127,127,0.9)'
  ctx.textAlign = 'center'
  for (let s = 0; s <= 60; s += 10) {
    ctx.fillText(s + 's', xOf(s * 1000), H - 12)
  }
  ctx.textAlign = 'left'
  ctx.fillText(`批大小（均值/200ms）`, padL + 2, top + 9)
  ctx.fillStyle = 'rgba(245,158,11,0.95)'
  ctx.fillText('平均延迟', padL + 150, top + 9)

  const k = Math.min(NBUCKET - 1, Math.floor(tNow / BUCKET))
  live.value = { t: tNow, bs: bucketBs.value[k], lat: bucketLat.value[k] }
}

function resize() {
  const c = canvas.value
  const w = wrap.value
  if (!c || !w) return
  dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = w.clientWidth
  const height = 190
  c.width = Math.round(width * dpr)
  c.height = Math.round(height * dpr)
  c.style.width = width + 'px'
  c.style.height = height + 'px'
  draw()
}

let ro: ResizeObserver | null = null
onMounted(() => {
  recompute()
  resize()
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
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">02 / demo4 · 实时模拟</div>
        <h3 class="simcard__title">动态批处理：系统最闲的时候，用户反而等得最久</h3>
      </div>
      <div class="simcard__badge" :class="{ 'is-busy': busy }">{{ busy ? '计算中…' : '与脚本一致' }}</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>策略</label>
        <div class="simcard__seg">
          <button v-for="p in PRESETS" :key="p.key" :class="{ 'is-on': selected === p.key }" @click="selected = p.key">
            {{ p.name }}
          </button>
        </div>
      </div>
      <div class="simcard__field">
        <label>平均 QPS <b>{{ qps }}</b> <span class="simcard__hint">（60 秒模拟 + 第 30~34 秒 3 倍尖峰）</span></label>
        <input v-model.number="qps" type="range" min="200" max="2000" step="50" />
      </div>
    </div>

    <div v-if="current" class="simcard__metrics">
      <div class="simcard__metric"><span>P50</span><b>{{ current.p50.toFixed(1) }} ms</b></div>
      <div class="simcard__metric"><span>P99</span><b>{{ current.p99.toFixed(1) }} ms</b></div>
      <div class="simcard__metric is-hot"><span>P999</span><b>{{ current.p999.toFixed(1) }} ms</b></div>
      <div class="simcard__metric"><span>最大</span><b>{{ current.max.toFixed(1) }} ms</b></div>
      <div class="simcard__metric"><span>平均批</span><b>{{ current.avgBs.toFixed(1) }}</b></div>
      <div class="simcard__metric"><span>GPU 利用率</span><b>{{ (current.util * 100).toFixed(1) }}%</b></div>
    </div>

    <div ref="wrap" class="simcard__chart" style="border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg)">
      <canvas ref="canvas" />
    </div>
    <div class="dbbar">
      <button class="simcard__btn" @click="playing ? pause() : start()">{{ playing ? '⏸ 暂停' : '▶ 播放' }}</button>
      <button class="simcard__btn" @click="restart">↺ 重播</button>
      <span class="simcard__seg simcard__seg--sm">
        <button v-for="s in SPEEDS" :key="s" :class="{ 'is-on': speed === s }" @click="speed = s">{{ s }}x</button>
      </span>
      <span class="dblive">t = {{ (live.t / 1000).toFixed(1) }}s · 当前批 ≈ {{ live.bs.toFixed(0) }} · 当前平均延迟 {{ live.lat.toFixed(1) }}ms</span>
    </div>

    <p class="simcard__read">
      固定批的致命问题：流量低谷时请求会被<b>饿死在队列里</b>等后来者凑批，而这恰恰是 GPU 最闲、
      本该最快响应的时候——完全是反的。切到「固定批 bs=256」看低谷期那根巨大的延迟尖峰。
      自适应让等待时间随队列积压走：积压少就不等（GPU 反正闲着），积压多才愿意凑大批，
      于是<b>低谷和高峰两段都不吃亏</b>。
    </p>

    <div class="simcard__block">
      <h4>六种策略对比</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>策略</th><th>P50</th><th>P99</th><th>P999</th><th>最大</th><th>平均批</th><th>GPU 利用率</th></tr>
          </thead>
          <tbody>
            <tr v-for="t in table" :key="t.key" :class="{ 'is-active': t.key === selected }" class="is-clickable" @click="selected = t.key">
              <td class="is-name">{{ t.name }}</td>
              <td>{{ t.p50.toFixed(1) }}</td>
              <td>{{ t.p99.toFixed(1) }}</td>
              <td class="is-hot">{{ t.p999.toFixed(1) }}</td>
              <td>{{ t.max.toFixed(1) }}</td>
              <td>{{ t.avgBs.toFixed(1) }}</td>
              <td>{{ (t.util * 100).toFixed(1) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="simcard__block">
      <h4>分段看：低谷 vs 尖峰</h4>
      <p class="simcard__sub">
        注意固定批策略的<b>低谷 P99 反而比尖峰更差</b>——尖峰期请求多，一眨眼就凑满一批；
        低谷期请求稀疏，先到的要干等后来者。
      </p>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>策略</th><th>低谷 P99</th><th>尖峰 P99</th><th>低谷 P999</th><th>尖峰 P999</th></tr>
          </thead>
          <tbody>
            <tr v-for="s in seg" :key="s.key" :class="{ 'is-active': s.key === selected }" class="is-clickable" @click="selected = s.key">
              <td class="is-name">{{ s.name }}</td>
              <td>{{ s.troughP99.toFixed(1) }} ms</td>
              <td>{{ s.spikeP99.toFixed(1) }} ms</td>
              <td class="is-hot">{{ s.troughP999.toFixed(1) }} ms</td>
              <td class="is-hot">{{ s.spikeP999.toFixed(1) }} ms</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <p class="simcard__foot">
      模型与 <code>02_compile/demo4_dynamic_batch.py</code> 一致：单 worker 离散事件模拟，
      单批耗时 = 2.0ms 固定 + 0.012ms × batch_size。
    </p>
  </div>
</template>

<style scoped>
.dbbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0 0;
  flex-wrap: wrap;
}
.dblive {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  margin-left: auto;
}
</style>
