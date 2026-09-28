<script setup lang="ts">
/**
 * GpuMemFragDemo —— 02_compile/demo3 的浏览器版。
 * 四种分配器跑同一段不定长分配/释放序列：碎片从哪来、谁先 OOM。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { genWorkload, makeAllocator, run, type AllocId, type Allocator, type WorkMode } from '../lib/alloc'
import { fmtMs } from '../lib/format'
import LineChart from './LineChart.vue'

const isClient = typeof window !== 'undefined'
const GPU_MB = 4096

const mode = ref<WorkMode>('skewed')
const watermark = ref(0.65)
const pageIdx = ref(0)
const PAGE_SIZES = [
  { label: '4MB', v: 4 },
  { label: '1MB', v: 1 },
  { label: '256KB', v: 0.25 },
]
const seed = ref(0)

interface Row {
  id: string
  name: string
  oom: number
  peak: number
  frag: number
  internal: number
  nfree: number
  intervals: Array<[number, number, boolean]>
}

const rows = ref<Row[]>([])
const scanning = ref(false)
const scanRows = ref<Array<{ wm: number; first: number; best: number; buddy: number; paged: number }>>([])

function buildAllocators(pageMb: number): Array<{ id: string; make: () => Allocator }> {
  return [
    { id: 'first', make: () => makeAllocator('first' as AllocId, GPU_MB, pageMb) },
    { id: 'best', make: () => makeAllocator('best' as AllocId, GPU_MB, pageMb) },
    { id: 'buddy', make: () => makeAllocator('buddy' as AllocId, GPU_MB, pageMb) },
    { id: 'paged', make: () => makeAllocator('paged' as AllocId, GPU_MB, pageMb) },
  ]
}

function compute() {
  scanning.value = true
  const pageMb = PAGE_SIZES[pageIdx.value].v
  const ops = genWorkload({ seed: seed.value, mode: mode.value, watermark: watermark.value })
  rows.value = buildAllocators(pageMb).map((a) => {
    const al = a.make()
    const r = run(al, ops)
    return {
      id: a.id,
      name: al.name,
      oom: r.oom,
      peak: r.peakUsed,
      frag: r.fragAvg,
      internal: r.internal,
      nfree: r.nfree,
      intervals: al.intervals(GPU_MB),
    }
  })

  const WMS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
  scanRows.value = WMS.map((wm) => {
    const o = genWorkload({ seed: seed.value, mode: mode.value, watermark: wm })
    const mk = (id: AllocId) => run(makeAllocator(id, GPU_MB, pageMb), o).oom
    return { wm, first: mk('first'), best: mk('best'), buddy: mk('buddy'), paged: mk('paged') }
  })
  scanning.value = false
  draw()
}

const scanSeries = computed(() => [
  { name: 'First-Fit', color: 'var(--sim-blue)', points: scanRows.value.map((r) => [r.wm, r.first] as [number, number]) },
  { name: 'Best-Fit', color: 'var(--sim-cyan)', points: scanRows.value.map((r) => [r.wm, r.best] as [number, number]) },
  { name: 'Buddy', color: 'var(--sim-red)', points: scanRows.value.map((r) => [r.wm, r.buddy] as [number, number]) },
  { name: `Paged/${PAGE_SIZES[pageIdx.value].label}`, color: 'var(--sim-green)', points: scanRows.value.map((r) => [r.wm, r.paged] as [number, number]) },
])

// ---------------------------------------------------------------- 地址空间可视化
const canvas = ref<HTMLCanvasElement | null>(null)
const wrap = ref<HTMLElement | null>(null)
let dpr = 1
let ro: ResizeObserver | null = null

function draw() {
  const c = canvas.value
  if (!c || !rows.value.length) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  const W = c.width / dpr
  const H = c.height / dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  const labelW = 92
  const laneH = 26
  const gap = 12
  const top = 8
  ctx.font = '11px ui-monospace, monospace'
  rows.value.forEach((r, i) => {
    const y = top + i * (laneH + gap)
    // 空闲底色
    ctx.fillStyle = 'rgba(127,127,127,0.10)'
    ctx.fillRect(labelW, y, W - labelW - 8, laneH)
    for (const [off, size, used] of r.intervals) {
      const x = labelW + (off / GPU_MB) * (W - labelW - 8)
      const w = Math.max(0.6, (size / GPU_MB) * (W - labelW - 8))
      ctx.fillStyle = used
        ? r.id === 'paged'
          ? 'rgba(16,185,129,0.85)'
          : 'rgba(59,130,246,0.85)'
        : 'rgba(127,127,127,0.10)'
      ctx.fillRect(x, y, w, laneH)
    }
    ctx.fillStyle = 'rgba(127,127,127,0.95)'
    ctx.textAlign = 'right'
    ctx.fillText(r.name.replace(/\(.*\)/, ''), labelW - 6, y + laneH / 2 + 4)
  })
  ctx.fillStyle = 'rgba(127,127,127,0.8)'
  ctx.textAlign = 'left'
  ctx.fillText('显存地址空间 0 → 4GB（灰 = 空闲空洞）', labelW, H - 4)
}

function resize() {
  const c = canvas.value
  const w = wrap.value
  if (!c || !w) return
  dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = w.clientWidth
  const height = 8 + rows.value.length * 38 + 14
  c.width = Math.round(width * dpr)
  c.height = Math.round(height * dpr)
  c.style.width = width + 'px'
  c.style.height = height + 'px'
  draw()
}

let timer: ReturnType<typeof setTimeout> | undefined
function schedule() {
  clearTimeout(timer)
  timer = setTimeout(() => {
    compute()
    resize()
  }, 60)
}
if (isClient) watch([mode, watermark, pageIdx, seed], schedule)

onMounted(() => {
  compute()
  resize()
  if (typeof ResizeObserver !== 'undefined' && wrap.value) {
    ro = new ResizeObserver(resize)
    ro.observe(wrap.value)
  }
})
onBeforeUnmount(() => {
  clearTimeout(timer)
  ro?.disconnect()
})
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">02 / demo3 · 实时模拟</div>
        <h3 class="simcard__title">显存碎片：为什么显存「还剩很多」却 OOM</h3>
      </div>
      <div class="simcard__badge" :class="{ 'is-busy': scanning }">{{ scanning ? '计算中…' : '与脚本一致' }}</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>请求长度分布</label>
        <div class="simcard__seg">
          <button :class="{ 'is-on': mode === 'skewed' }" @click="mode = 'skewed'">长尾（真实推荐：序列长短不一）</button>
          <button :class="{ 'is-on': mode === 'uniform' }" @click="mode = 'uniform'">均匀（对照组）</button>
        </div>
      </div>
      <div class="simcard__field">
        <label>存活数据水位 <b>{{ (watermark * 100).toFixed(0) }}%</b> <span class="simcard__hint">（一个理想分配器应当全程 0 OOM）</span></label>
        <input v-model.number="watermark" type="range" min="0.55" max="0.95" step="0.01" />
      </div>
      <div class="simcard__row">
        <div class="simcard__field">
          <label>分页大小</label>
          <div class="simcard__seg">
            <button v-for="(p, i) in PAGE_SIZES" :key="p.label" :class="{ 'is-on': pageIdx === i }" @click="pageIdx = i">
              {{ p.label }}
            </button>
          </div>
        </div>
        <button class="simcard__btn" @click="seed++">重新生成分配序列 ↻</button>
      </div>
    </div>

    <div class="simcard__block">
      <h4>四种分配器在同一段序列上的表现</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>分配器</th><th>OOM 次数</th><th>峰值占用</th><th>外部碎片率</th><th>内部碎片</th><th>空闲块数</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in rows" :key="r.id">
              <td class="is-name">{{ r.name }}</td>
              <td :class="{ 'is-hot': r.oom > 0 }">{{ r.oom }}</td>
              <td>{{ r.peak.toFixed(0) }} MB</td>
              <td>{{ (r.frag * 100).toFixed(1) }}%</td>
              <td :class="{ 'is-hot': r.internal > 0.15 }">{{ (r.internal * 100).toFixed(1) }}%</td>
              <td>{{ r.nfree }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div ref="wrap" class="simcard__chart" style="margin-top: 14px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg)">
        <canvas ref="canvas" />
      </div>
      <p class="simcard__read">
        灰色是<b>空闲但用不上</b>的空洞——外部碎片。First-Fit / Best-Fit 在跟它搏斗；
        Buddy 用「向上取整到 2 的幂」换快速合并，代价是约 30% 的内部碎片，所以最早垮；
        Paged 把外部碎片<b>直接降为 0</b>（任何一页都能给任何请求用），只剩最后一页的零头。
      </p>
    </div>

    <div class="simcard__block">
      <h4>水位扫描：显存用到几成时，碎片开始真正伤人</h4>
      <p class="simcard__sub">
        每个非零 OOM 都**不是**因为显存真的不够——存活数据始终在水位以下。
        所以每一个非零数字都是碎片付出的代价。
      </p>
      <LineChart
        :series="scanSeries"
        x-label="存活数据水位"
        y-label="OOM 次数"
        :x-ticks="scanRows.map((r) => ({ v: r.wm, label: (r.wm * 100).toFixed(0) + '%' }))"
        :y-format="(v: number) => String(Math.round(v))"
      />
      <p class="simcard__read">
        Paged/4MB 在高水位同样 OOM，但原因完全不同：它没有外部碎片，垮掉是因为 4MB 的页对中位数仅 ~5MB
        的请求太粗。把页减小到 1MB / 256KB，内部碎片随之下降，抗压水位显著提高。
        结论不是「分页万能」，而是<b>分页把难以控制的外部碎片，换成了可以用页大小这一个旋钮精确调节的内部碎片</b>。
      </p>
    </div>

    <p class="simcard__foot">
      模型与 <code>02_compile/demo3_gpu_mem_frag.py</code> 一致：4096MB 显存、20000 次不定长分配/释放；
      随机数发生器不同，数值有统计波动。
    </p>
  </div>
</template>
