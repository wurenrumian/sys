<script setup lang="ts">
/**
 * AllReduceDemo —— 03_network/demo2 的浏览器版。
 * alpha-beta 模型：T = 轮数 × 单跳延迟 + 传输字节 / 带宽。
 */
import { computed, ref } from 'vue'
import { ALGOS, LINKS, hierarchical, linkById } from '../lib/allreduce'
import { fmtBytes, fmtMs } from '../lib/format'
import LineChart from './LineChart.vue'

const N_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
const N_BUTTONS = [8, 64, 256, 1024, 4096]
const D_LIST = [
  { label: '1KB', v: 1e3 },
  { label: '1MB', v: 1e6 },
  { label: '10MB', v: 1e7 },
  { label: '100MB', v: 1e8 },
  { label: '1GB', v: 1e9 },
]

const n = ref(1024)
const D = ref(1e9)
const linkId = ref('ib')
const showHier = ref(false)

const link = computed(() => linkById(linkId.value))
const times = computed(() =>
  ALGOS.map((a) => ({ ...a, t: a.fn(n.value, D.value, link.value.bw, link.value.lat) })),
)
const bestT = computed(() => Math.min(...times.value.map((x) => x.t)))
const ringT = computed(() => times.value[1].t)
const hdT = computed(() => times.value[3].t)
const ringBwShare = computed(() => {
  const bwTerm = (2 * (n.value - 1) / n.value) * D.value / (link.value.bw * 1e9)
  return bwTerm / ringT.value
})

const scaling = computed(() => {
  const s = N_LIST.map((nn) => ({
    n: nn,
    ring: ALGOS[1].fn(nn, D.value, link.value.bw, link.value.lat),
    tree: ALGOS[2].fn(nn, D.value, link.value.bw, link.value.lat),
    hd: ALGOS[3].fn(nn, D.value, link.value.bw, link.value.lat),
  }))
  return [
    { name: 'Ring', color: 'var(--sim-blue)', points: s.map((x) => [x.n, x.ring] as [number, number]) },
    { name: '二项树', color: 'var(--sim-red)', points: s.map((x) => [x.n, x.tree] as [number, number]) },
    { name: '递归折半HD', color: 'var(--sim-green)', points: s.map((x) => [x.n, x.hd] as [number, number]) },
  ]
})

const crossing = computed(() => {
  const ds: number[] = []
  for (let e = 3; e <= 9; e += 0.3) ds.push(Math.pow(10, e))
  return [
    { name: 'Ring', color: 'var(--sim-blue)', points: ds.map((d) => [d, ALGOS[1].fn(n.value, d, link.value.bw, link.value.lat)] as [number, number]) },
    { name: '二项树', color: 'var(--sim-red)', points: ds.map((d) => [d, ALGOS[2].fn(n.value, d, link.value.bw, link.value.lat)] as [number, number]) },
    { name: '递归折半HD', color: 'var(--sim-green)', points: ds.map((d) => [d, ALGOS[3].fn(n.value, d, link.value.bw, link.value.lat)] as [number, number]) },
  ]
})

const hier = computed(() =>
  hierarchical({
    machines: 128,
    gpusPerMachine: 8,
    D: D.value,
    intra: linkById('nvlink'),
    inter: linkById('ib'),
  }),
)

function barWidth(t: number) {
  // 对数长度，避免 Naive 把其他算法压成一条线
  return 8 + (Math.log10(Math.max(t, bestT.value)) - Math.log10(bestT.value)) * 24
}
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">03 / demo2 · 实时模型</div>
        <h3 class="simcard__title">AllReduce：为什么「千卡之后通信效率变低」是可以算出来的</h3>
      </div>
      <div class="simcard__badge">解析模型 · 与脚本一致</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>卡数 <b>{{ n.toLocaleString() }}</b></label>
        <div class="simcard__seg">
          <button v-for="nn in N_BUTTONS" :key="nn" :class="{ 'is-on': n === nn }" @click="n = nn">
            {{ nn }}
          </button>
        </div>
      </div>
      <div class="simcard__field">
        <label>数据量（梯度大小）<b>{{ fmtBytes(D) }}</b></label>
        <div class="simcard__seg">
          <button v-for="d in D_LIST" :key="d.label" :class="{ 'is-on': D === d.v }" @click="D = d.v">
            {{ d.label }}
          </button>
        </div>
      </div>
      <div class="simcard__field">
        <label>链路</label>
        <div class="simcard__seg">
          <button v-for="l in LINKS" :key="l.id" :class="{ 'is-on': linkId === l.id }" @click="linkId = l.id">
            {{ l.name }} · {{ l.bw }}GB/s
          </button>
        </div>
      </div>
    </div>

    <div class="simcard__metrics">
      <div class="simcard__metric"><span>最快算法</span><b>{{ times.find((t) => t.t === bestT)?.name }}</b></div>
      <div class="simcard__metric"><span>Ring 耗时</span><b>{{ fmtMs(ringT * 1e3) }}</b></div>
      <div class="simcard__metric"><span>递归折半 HD</span><b>{{ fmtMs(hdT * 1e3) }}</b></div>
      <div class="simcard__metric"><span>Ring 带宽项占比</span><b>{{ (ringBwShare * 100).toFixed(1) }}%</b></div>
    </div>

    <div class="simcard__block">
      <h4>当前配置下四种算法的耗时</h4>
      <p class="simcard__sub">
        条长按对数刻度（否则 Naive 会把其他算法压成一条线）；具体数值见右侧。
        Naive 的中心节点要收发 (N−1)·D 字节，完全不可扩展；Ring 单卡传输量趋近 2D（与 N 无关），
        但要走 2(N−1) 轮，延迟项随 N 线性增长。
      </p>
      <div class="arbars">
        <div v-for="t in times" :key="t.id" class="arbars__row">
          <span class="arbars__name">{{ t.name }}</span>
          <span class="arbars__bar" :style="{ width: barWidth(t.t) + 'px', background: t.id === 'ring' ? 'var(--sim-blue)' : t.id === 'tree' ? 'var(--sim-red)' : t.id === 'hd' ? 'var(--sim-green)' : 'var(--vp-c-text-3)' }" />
          <span class="arbars__val">{{ fmtMs(t.t * 1e3) }}</span>
        </div>
      </div>
    </div>

    <div class="simcard__block">
      <h4>规模扩展：同样的数据量，卡数越多越慢</h4>
      <p class="simcard__sub">固定数据量、固定链路，卡片数从 8 到 4096。纵轴对数。</p>
      <LineChart
        :series="scaling"
        x-label="卡数"
        y-label="通信耗时（对数）"
        x-log
        y-log
        :x-ticks="N_LIST.map((v) => ({ v, label: String(v) }))"
        :y-format="(v: number) => fmtMs(v * 1e3, 0)"
      />
    </div>

    <div class="simcard__block">
      <h4>Ring vs Tree 的交叉点：取决于数据量</h4>
      <p class="simcard__sub">
        小数据量被延迟支配（树赢，轮数少）；大数据量被带宽支配（Ring 赢，单卡传输量与 N 无关）。
        横轴对数，纵轴对数。
      </p>
      <LineChart
        :series="crossing"
        x-label="数据量"
        y-label="通信耗时（对数）"
        x-log
        y-log
        :x-format="(v: number) => fmtBytes(v)"
        :y-format="(v: number) => fmtMs(v * 1e3, 0)"
      />
    </div>

    <div class="simcard__block">
      <h4>分层 AllReduce：把慢链路上的数据量降到 1/8</h4>
      <p class="simcard__sub">
        1024 卡 = 128 机 × 8 卡。先机内 reduce-scatter，机间只对 D/8 做 AllReduce，最后机内 all-gather。
      </p>
      <div class="simcard__metrics">
        <div class="simcard__metric"><span>扁平 Ring（全走 IB）</span><b>{{ fmtMs(hier.flat * 1e3) }}</b></div>
        <div class="simcard__metric is-hot"><span>分层 AllReduce</span><b>{{ fmtMs(hier.hier * 1e3) }}</b></div>
        <div class="simcard__metric"><span>加速比</span><b>{{ (hier.flat / hier.hier).toFixed(2) }}x</b></div>
      </div>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <tbody>
            <tr><td class="is-name">机内 reduce-scatter (NVLink)</td><td>{{ fmtMs(hier.rs * 1e3) }}</td></tr>
            <tr><td class="is-name">机间 AllReduce (IB，仅 D/8)</td><td>{{ fmtMs(hier.inter * 1e3) }}</td></tr>
            <tr><td class="is-name">机内 all-gather (NVLink)</td><td>{{ fmtMs(hier.ag * 1e3) }}</td></tr>
          </tbody>
        </table>
      </div>
      <p class="simcard__read">
        收益的全部来源：机间（慢链路）上的数据量从 D 降到了 D/8，剩下的规约工作交给快
        {{ Math.round(450 / 25) }} 倍的机内 NVLink。
      </p>
    </div>

    <p class="simcard__foot">
      模型：<code>T = 通信轮数 × 单跳延迟 + 传输字节 / 带宽</code>，与
      <code>03_network/demo2_allreduce.py</code> 完全一致。它只是**下界**，假设了完美带宽、
      零拥塞、零抖动。
    </p>
  </div>
</template>

<style scoped>
.arbars {
  display: grid;
  gap: 8px;
}
.arbars__row {
  display: grid;
  grid-template-columns: 130px 1fr 90px;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.arbars__name {
  color: var(--vp-c-text-2);
}
.arbars__bar {
  height: 12px;
  border-radius: 4px;
  min-width: 4px;
  transition: width 0.25s;
}
.arbars__val {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 640px) {
  .arbars__row {
    grid-template-columns: 96px 1fr 78px;
    font-size: 12px;
  }
}
</style>
