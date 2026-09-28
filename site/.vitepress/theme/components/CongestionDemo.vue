<script setup lang="ts">
/**
 * CongestionDemo —— 03_network/demo3 的浏览器版。
 * 流体近似：队列深度 / 链路速率 = 排队延迟，所以低队列就是低尾延迟。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { CCS, DCTCP_K, LINK_PKT_PER_MS, QUEUE_CAP, SIM_MS, simulateCC, type CCId, type CongResult } from '../lib/congestion'
import { fmtMs } from '../lib/format'
import LineChart from './LineChart.vue'

const isClient = typeof window !== 'undefined'

const cc = ref<CCId>('reno')
const k = ref(DCTCP_K)
const nFlows = ref(4)
const busy = ref(false)
const result = ref<CongResult | null>(null)
const compare = ref<Array<{ id: CCId; name: string; r: CongResult }>>([])

let timer: ReturnType<typeof setTimeout> | undefined
function recompute() {
  busy.value = true
  clearTimeout(timer)
  timer = setTimeout(() => {
    result.value = simulateCC(cc.value, { nFlows: nFlows.value, k: k.value })
    compare.value = CCS.map((c) => ({ ...c, r: simulateCC(c.id, { nFlows: nFlows.value, k: k.value }) }))
    busy.value = false
    restart()
  }, 40)
}
if (isClient) watch([cc, k, nFlows], recompute)

// ---------------------------------------------------------------- K 扫描
const kScan = computed(() => {
  const ks = [2, 5, 10, 20, 50, 100, 200, 400]
  const pts = ks.map((kk) => {
    const r = simulateCC('dctcp', { nFlows: nFlows.value, k: kk })
    return { k: kk, delay: r.delayP99, util: r.util }
  })
  return {
    rows: pts,
    series: [
      { name: 'P99 延迟', color: 'var(--sim-red)', points: pts.map((p) => [p.k, p.delay] as [number, number]) },
    ] as any,
  }
})

// ---------------------------------------------------------------- 动画
const canvas = ref<HTMLCanvasElement | null>(null)
const wrap = ref<HTMLElement | null>(null)
const playing = ref(true)
const progress = ref(0)
const speed = ref(1)
const SPEEDS = [0.5, 1, 2, 4]
const live = ref({ t: 0, q: 0, cwnd: 0, delay: 0 })

let raf = 0
let prevTs = 0
let dpr = 1

function frame(ts: number) {
  if (!playing.value) return
  if (!prevTs) prevTs = ts
  const dt = (ts - prevTs) / 1000
  prevTs = ts
  progress.value += dt * 900 * speed.value
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
  const r = result.value
  if (!c || !r) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  const W = c.width / dpr
  const H = c.height / dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  const pad = 56
  const top = 16
  const bot = H - 18

  // 队列容量参考
  const yq = (q: number) => bot - (q / QUEUE_CAP) * (bot - top)
  ctx.strokeStyle = 'rgba(127,127,127,0.35)'
  ctx.setLineDash([4, 3])
  ctx.beginPath()
  ctx.moveTo(pad, yq(QUEUE_CAP))
  ctx.lineTo(W - 12, yq(QUEUE_CAP))
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(pad, yq(k.value))
  ctx.lineTo(W - 12, yq(k.value))
  ctx.stroke()
  ctx.setLineDash([])
  ctx.font = '10px ui-monospace, monospace'
  ctx.fillStyle = 'rgba(127,127,127,0.95)'
  ctx.textAlign = 'left'
  ctx.fillText('队列容量 500 包', pad + 4, yq(QUEUE_CAP) - 3)
  ctx.fillText(`K = ${k.value} 包`, pad + 4, yq(k.value) - 3)

  const xOf = (t: number) => pad + (t / SIM_MS) * (W - pad - 12)
  const tNow = progress.value

  // 队列深度（面积）
  ctx.beginPath()
  ctx.moveTo(xOf(0), bot)
  for (let t = 0; t <= tNow; t++) ctx.lineTo(xOf(t), yq(r.qSeries[t]))
  ctx.lineTo(xOf(tNow), bot)
  ctx.closePath()
  ctx.fillStyle = 'rgba(245,158,11,0.35)'
  ctx.fill()
  ctx.strokeStyle = 'rgba(245,158,11,0.95)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  for (let t = 0; t <= tNow; t++) {
    const x = xOf(t)
    const y = yq(r.qSeries[t])
    t ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
  }
  ctx.stroke()
  ctx.lineWidth = 1

  // 队列平均线
  ctx.strokeStyle = 'rgba(239,68,68,0.8)'
  ctx.beginPath()
  ctx.moveTo(pad, yq(r.qAvg))
  ctx.lineTo(W - 12, yq(r.qAvg))
  ctx.stroke()

  // 播放头
  const px = xOf(tNow)
  ctx.strokeStyle = '#ef4444'
  ctx.beginPath()
  ctx.moveTo(px, top)
  ctx.lineTo(px, bot)
  ctx.stroke()

  const qNow = r.qSeries[Math.min(r.qSeries.length - 1, Math.floor(tNow))]
  live.value = { t: tNow, q: qNow, cwnd: r.cwndSeries[Math.min(r.cwndSeries.length - 1, Math.floor(tNow))], delay: 1 + qNow / LINK_PKT_PER_MS }

  ctx.fillStyle = 'rgba(127,127,127,0.95)'
  ctx.textAlign = 'right'
  ctx.fillText('队列深度', pad - 6, top + 10)
  ctx.fillText(`${QUEUE_CAP} 包`, pad - 6, yq(QUEUE_CAP) + 3)
  ctx.fillText('0', pad - 6, bot)
}

function resize() {
  const c = canvas.value
  const w = wrap.value
  if (!c || !w) return
  dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = w.clientWidth
  const height = 170
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
        <div class="simcard__eyebrow">03 / demo3 · 实时模拟</div>
        <h3 class="simcard__title">拥塞控制：为什么一个算法同时服务不了「低延迟」和「高吞吐」</h3>
      </div>
      <div class="simcard__badge" :class="{ 'is-busy': busy }">{{ busy ? '计算中…' : '流体近似 · 与脚本一致' }}</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>拥塞控制算法</label>
        <div class="simcard__seg">
          <button v-for="c in CCS" :key="c.id" :class="{ 'is-on': cc === c.id }" @click="cc = c.id">
            {{ c.name }}
          </button>
        </div>
      </div>
      <div class="simcard__field">
        <label>DCTCP 的 ECN 标记阈值 K <b>{{ k }} 包</b> <span class="simcard__hint">（只影响 DCTCP）</span></label>
        <input v-model.number="k" type="range" min="2" max="400" step="1" />
      </div>
    </div>

    <div v-if="result" class="simcard__metrics">
      <div class="simcard__metric"><span>链路利用率</span><b>{{ (result.util * 100).toFixed(1) }}%</b></div>
      <div class="simcard__metric"><span>平均队列</span><b>{{ result.qAvg.toFixed(0) }} 包</b></div>
      <div class="simcard__metric is-hot"><span>P99 队列</span><b>{{ result.qP99.toFixed(0) }} 包</b></div>
      <div class="simcard__metric"><span>平均延迟</span><b>{{ result.delayAvg.toFixed(2) }} ms</b></div>
      <div class="simcard__metric is-hot"><span>P99 延迟</span><b>{{ result.delayP99.toFixed(2) }} ms</b></div>
      <div class="simcard__metric"><span>丢包量</span><b>{{ result.drops.toFixed(0) }}</b></div>
    </div>

    <div ref="wrap" class="simcard__chart congise" style="border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg)">
      <canvas ref="canvas" />
    </div>
    <div class="congbar">
      <button class="simcard__btn" @click="playing ? pause() : start()">{{ playing ? '⏸ 暂停' : '▶ 播放' }}</button>
      <button class="simcard__btn" @click="restart">↺ 重播</button>
      <span class="simcard__seg simcard__seg--sm">
        <button v-for="s in SPEEDS" :key="s" :class="{ 'is-on': speed === s }" @click="speed = s">{{ s }}x</button>
      </span>
      <span class="conglive">
        t = {{ live.t.toFixed(0) }}ms · 队列 {{ live.q.toFixed(0) }} 包 · 平均 cwnd {{ live.cwnd.toFixed(1) }} 包 ·
        当前 RTT {{ live.delay.toFixed(2) }}ms
      </span>
    </div>

    <p class="simcard__read">
      队列深度 ÷ 链路速率 = 排队延迟，所以 <b>队列深度和尾延迟几乎是一回事</b>。
      Reno 必须把队列填满、丢包了才知道拥塞，于是队列常年很深（bufferbloat）；
      DCTCP 用 ECN 在队列变深之前就拿到信号，队列稳在 K 附近；BBR 式直接控制 BDP，把队列留空。
      <b>但所有算法的链路利用率都在 98% 以上</b> —— 真正的取舍不在吞吐，而在队列。
    </p>

    <div class="simcard__block">
      <h4>同一份负载下三种算法</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>算法</th><th>利用率</th><th>平均队列</th><th>P99 队列</th><th>平均延迟</th><th>P99 延迟</th></tr>
          </thead>
          <tbody>
            <tr v-for="c in compare" :key="c.id" :class="{ 'is-active': c.id === cc }">
              <td class="is-name">{{ c.name }}</td>
              <td>{{ (c.r.util * 100).toFixed(1) }}%</td>
              <td>{{ c.r.qAvg.toFixed(0) }} 包</td>
              <td>{{ c.r.qP99.toFixed(0) }} 包</td>
              <td>{{ c.r.delayAvg.toFixed(2) }} ms</td>
              <td class="is-hot">{{ c.r.delayP99.toFixed(2) }} ms</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="simcard__block">
      <h4>DCTCP 的 K 怎么选：模型里调小几乎免费，现实中不行</h4>
      <p class="simcard__sub">
        这个流体模型没建模「突发吸收」和「反馈时延」，所以 K 越小延迟越好、利用率却掉不下来。
        真实场景里 K 太小会让突发直接变丢包、ECN 来不及反馈就溢出。
      </p>
      <LineChart
        :series="kScan.series"
        x-label="ECN 阈值 K（包）"
        y-label="P99 延迟 (ms)"
        :y-format="(v: number) => v.toFixed(1)"
      />
      <p class="simcard__read">
        RPC 要浅队列（低延迟），训练大流要深队列（吸收突发、保吞吐）——
        <b>同一个队列没法同时既浅又深</b>。出路是给两类流量分配不同的队列，
        而这又要求网络能区分它们（回到 demo1 和跨层信令的问题）。
      </p>
    </div>

    <p class="simcard__foot">
      模型与 <code>03_network/demo3_congestion.py</code> 一致：按 ms 推进的流体近似，
      能复现队列深度与利用率的基本动力学，但具体数值不代表真实测量。
    </p>
  </div>
</template>

<style scoped>
.congise canvas {
  display: block;
}
.congbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0 0;
  flex-wrap: wrap;
}
.conglive {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  margin-left: auto;
}
</style>
