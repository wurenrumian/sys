<script setup lang="ts">
/**
 * 通用折线/散点图（SVG）。被多个 demo 复用，避免每处重画坐标轴。
 * 支持线性/对数轴、参考线、高亮点、tooltip。
 */
import { computed } from 'vue'

export interface Series {
  name: string
  color: string
  points: Array<[number, number]>
  dashed?: boolean
  dots?: boolean
  area?: boolean
}

const props = withDefaults(
  defineProps<{
    series: Series[]
    xLabel?: string
    yLabel?: string
    xLog?: boolean
    yLog?: boolean
    xTicks?: Array<{ v: number; label: string }>
    yFormat?: (v: number) => string
    xFormat?: (v: number) => string
    refLines?: Array<{ x?: number; y?: number }>
    markers?: Array<{ x: number; y: number; color?: string }>
    width?: number
    height?: number
    yMin?: number
    yMax?: number
    yTicks?: number[]
  }>(),
  { width: 660, height: 280, xFormat: undefined, yFormat: undefined },
)

const padL = 56
const padR = 16
const padT = 16
const padB = 42

const tx = (v: number) => (props.xLog ? Math.log10(Math.max(v, 1e-12)) : v)
const ty = (v: number) => (props.yLog ? Math.log10(Math.max(v, 1e-12)) : v)

const all = computed(() => props.series.flatMap((s) => s.points))
const hasData = computed(() => all.value.length > 0)
const dx = computed(() => {
  if (!all.value.length) return [0, 1] as [number, number]
  const xs = all.value.map((p) => tx(p[0]))
  return [Math.min(...xs), Math.max(...xs)] as [number, number]
})
const dy = computed(() => {
  if (!all.value.length) return [0, 1] as [number, number]
  const ys = all.value.map((p) => ty(p[1]))
  const lo = props.yMin !== undefined ? props.yMin : Math.min(...ys)
  const hi = props.yMax !== undefined ? props.yMax : Math.max(...ys)
  return [lo, hi] as [number, number]
})

function sx(v: number) {
  const [lo, hi] = dx.value
  const f = hi === lo ? 0.5 : (tx(v) - lo) / (hi - lo)
  return padL + f * (props.width - padL - padR)
}
function sy(v: number) {
  const [lo, hi] = dy.value
  const f = hi === lo ? 0.5 : (ty(v) - lo) / (hi - lo)
  return props.height - padB - f * (props.height - padT - padB)
}

function pathOf(s: Series) {
  return s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ')
}
function areaOf(s: Series) {
  const pts = s.points
  if (!pts.length) return ''
  const base = props.height - padB
  return (
    `M${sx(pts[0][0])},${base} ` +
    pts.map((p) => `L${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ') +
    ` L${sx(pts[pts.length - 1][0])},${base} Z`
  )
}

const xTickList = computed(() => {
  if (props.xTicks) return props.xTicks
  const [lo, hi] = dx.value
  const out: Array<{ v: number; label: string }> = []
  if (props.xLog) {
    // 对数轴：整幂次刻度，避免出现 15.8KB 这种怪刻度
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
      const lv = e
      if (lv < lo - 1e-9 || lv > hi + 1e-9) continue
      const v = Math.pow(10, e)
      out.push({ v, label: props.xFormat ? props.xFormat(v) : String(v) })
    }
    return out
  }
  const n = 5
  for (let i = 0; i <= n; i++) {
    const t = lo + ((hi - lo) * i) / n
    out.push({ v: t, label: props.xFormat ? props.xFormat(t) : String(round(t)) })
  }
  return out
})

/** 靠边的刻度标签改变对齐方式，避免被裁切 */
function xAnchor(v: number): 'start' | 'middle' | 'end' {
  const x = sx(v)
  if (x < padL + 16) return 'start'
  if (x > props.width - padR - 20) return 'end'
  return 'middle'
}
function round(v: number) {
  if (Math.abs(v) >= 1000) return Math.round(v / 100) * 100
  if (Math.abs(v) >= 10) return Math.round(v)
  return Math.round(v * 100) / 100
}

const yTickList = computed(() => {
  if (props.yTicks) return props.yTicks
  const [lo, hi] = dy.value
  if (props.yLog) {
    const out: number[] = []
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) out.push(Math.pow(10, e))
    return out
  }
  const out: number[] = []
  const n = 4
  for (let i = 0; i <= n; i++) out.push(lo + ((hi - lo) * i) / n)
  return out
})
function yLabelOf(v: number) {
  return props.yFormat ? props.yFormat(v) : String(round(v))
}

const uid = Math.random().toString(36).slice(2, 8)
</script>

<template>
  <div class="lchart">
    <svg :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="xMidYMid meet">
      <defs>
        <clipPath :id="'clip-' + uid">
          <rect :x="padL" :y="padT" :width="width - padL - padR" :height="height - padT - padB" />
        </clipPath>
      </defs>

      <!-- y 轴网格 -->
      <g>
        <line
          v-for="(v, i) in yTickList"
          :key="'y' + i"
          :x1="padL"
          :x2="width - padR"
          :y1="sy(v)"
          :y2="sy(v)"
          class="grid"
        />
        <text
          v-for="(v, i) in yTickList"
          :key="'yl' + i"
          :x="padL - 6"
          :y="sy(v) + 3"
          text-anchor="end"
          class="axis"
        >
          {{ yLabelOf(v) }}
        </text>
      </g>

      <!-- x 轴刻度 -->
      <g>
        <text
          v-for="(t, i) in xTickList"
          :key="'xl' + i"
          :x="sx(t.v)"
          :y="height - padB + 16"
          :text-anchor="xAnchor(t.v)"
          class="axis"
        >
          {{ t.label }}
        </text>
      </g>

      <!-- 参考线 -->
      <g>
        <line
          v-for="(r, i) in refLines || []"
          :key="'r' + i"
          :x1="r.x !== undefined ? sx(r.x) : padL"
          :x2="r.x !== undefined ? sx(r.x) : width - padR"
          :y1="r.y !== undefined ? sy(r.y) : padT"
          :y2="r.y !== undefined ? sy(r.y) : height - padB"
          class="refline"
        />
      </g>

      <g :clip-path="`url(#clip-${uid})`">
        <template v-for="(s, si) in series" :key="'s' + si">
          <path v-if="s.area" :d="areaOf(s)" :fill="s.color" opacity="0.12" />
          <path
            :d="pathOf(s)"
            class="line"
            :stroke="s.color"
            :stroke-dasharray="s.dashed ? '5 4' : undefined"
          />
          <circle
            v-for="(p, pi) in s.dots === false ? [] : s.points"
            :key="'d' + si + '-' + pi"
            :cx="sx(p[0])"
            :cy="sy(p[1])"
            r="2.8"
            :fill="s.color"
            class="dot"
          >
            <title>{{ s.name }}: x={{ xFormat ? xFormat(p[0]) : p[0] }}, y={{ yLabelOf(p[1]) }}</title>
          </circle>
        </template>

        <circle
          v-for="(m, mi) in markers || []"
          :key="'m' + mi"
          :cx="sx(m.x)"
          :cy="sy(m.y)"
          r="4.5"
          :fill="m.color || 'var(--sim-red)'"
          class="marker"
        />
      </g>

      <text
        v-if="!hasData"
        :x="(padL + width - padR) / 2"
        :y="(padT + height - padB) / 2"
        text-anchor="middle"
        class="axis"
      >
        计算中…
      </text>

      <text
        :x="(padL + width - padR) / 2"
        :y="height - 4"
        text-anchor="middle"
        class="axis axis--title"
      >
        {{ xLabel }}
      </text>
      <text
        :x="14"
        :y="(padT + height - padB) / 2"
        text-anchor="middle"
        class="axis axis--title"
        :transform="`rotate(-90 14 ${(padT + height - padB) / 2})`"
      >
        {{ yLabel }}
      </text>
    </svg>
    <div v-if="series.length > 1" class="lchart__legend simcard__legend">
      <span v-for="(s, i) in series" :key="'lg' + i">
        <i class="simcard__sw" :style="{ background: s.color }" />{{ s.name }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.lchart {
  width: 100%;
}
.lchart svg {
  width: 100%;
  height: auto;
  display: block;
}
.grid {
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}
.axis {
  fill: var(--vp-c-text-3);
  font-size: 11px;
  font-family: var(--vp-font-family-mono);
}
.axis--title {
  font-family: inherit;
  font-size: 11.5px;
}
.line {
  fill: none;
  stroke-width: 2;
}
.refline {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.dot {
  stroke: var(--vp-c-bg);
  stroke-width: 1.2;
}
.marker {
  stroke: var(--vp-c-bg);
  stroke-width: 1.5;
}
</style>
