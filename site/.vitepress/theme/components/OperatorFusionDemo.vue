<script setup lang="ts">
/**
 * OperatorFusionDemo —— 02_compile/demo2 的交互式分析。
 * d = relu(a*b + c) * s：算术量完全相同，差的只是访存量。
 */
import { computed, ref } from 'vue'
import { fmtMs } from '../lib/format'
import LineChart from './LineChart.vue'

const N = ref(8_000_000)
const N_LIST = [100_000, 1_000_000, 8_000_000, 32_000_000]

interface Impl {
  name: string
  passes: number
  kernels: number
  color: string
}
const IMPLS: Impl[] = [
  { name: '未融合', passes: 8, kernels: 4, color: 'var(--sim-red)' },
  { name: 'numpy out=', passes: 7, kernels: 4, color: 'var(--sim-amber)' },
  { name: '分块融合（理论下限）', passes: 4, kernels: 1, color: 'var(--sim-green)' },
]

const rows = computed(() =>
  IMPLS.map((im) => {
    const bytes = N.value * 4 * im.passes
    const flops = N.value * 3
    return { ...im, bytes, intensity: flops / bytes }
  }),
)
const maxBytes = computed(() => Math.max(...rows.value.map((r) => r.bytes)))

// 实测：规模扩展（来自真实输出）
const SCALE = [
  { n: 0.1, un: 0.07, out: 0.05, blk: 0.08, sp: 0.89 },
  { n: 1, un: 1.34, out: 0.59, blk: 0.92, sp: 1.46 },
  { n: 8, un: 15.83, out: 12.86, blk: 10.51, sp: 1.51 },
  { n: 32, un: 94.39, out: 60.83, blk: 48.95, sp: 1.93 },
]

// 实测：块大小扫描（N=8M，未融合基线 14.29ms）
const BLOCKS = [
  { kb: 16, t: 22.91, fits: 'L1/L2', blocks: 7813 },
  { kb: 128, t: 10.49, fits: 'L1/L2', blocks: 977 },
  { kb: 1024, t: 9.4, fits: 'L3', blocks: 123 },
  { kb: 4096, t: 9.71, fits: 'L3', blocks: 31 },
  { kb: 16384, t: 10.21, fits: 'L3', blocks: 8 },
  { kb: 125000, t: 14.31, fits: '不进', blocks: 1 },
]
const blockChart = computed(() => [
  {
    name: '分块融合 vs 未融合',
    color: 'var(--sim-green)',
    points: BLOCKS.map((b) => [b.kb, 14.29 / b.t] as [number, number]),
  },
])
const scaleChart = computed(() => [
  { name: '未融合', color: 'var(--sim-red)', points: SCALE.map((s) => [s.n, s.un] as [number, number]) },
  { name: 'numpy out=', color: 'var(--sim-amber)', points: SCALE.map((s) => [s.n, s.out] as [number, number]) },
  { name: '分块融合', color: 'var(--sim-green)', points: SCALE.map((s) => [s.n, s.blk] as [number, number]) },
])
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">02 / demo2 · 交互式分析</div>
        <h3 class="simcard__title">算子融合：算术量没变，访存量差一倍</h3>
      </div>
      <div class="simcard__badge">访存模型 + 本机实测</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>数组规模 N <b>{{ N.toLocaleString() }}</b>（float32）</label>
        <div class="simcard__seg">
          <button v-for="n in N_LIST" :key="n" :class="{ 'is-on': N === n }" @click="N = n">
            {{ n >= 1e6 ? n / 1e6 + 'M' : n / 1e3 + 'K' }}
          </button>
        </div>
        <p class="simcard__hint" style="margin: 10px 0 0">
          `d = relu(a*b + c) * s` 的算术量只有 3N FLOP；未融合实现每步都要产生一个全尺寸临时数组。
        </p>
      </div>
    </div>

    <div class="simcard__block">
      <h4>访存量：融合到底省了什么</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>实现</th><th>kernel 数</th><th>内存遍历</th><th>访存字节</th><th>算术强度</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in rows" :key="r.name">
              <td class="is-name">{{ r.name }}</td>
              <td>{{ r.kernels }}</td>
              <td>{{ r.passes }}</td>
              <td>{{ (r.bytes / 1e6).toFixed(0) }} MB</td>
              <td>{{ r.intensity.toFixed(2) }} F/B</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="ofbars">
        <div v-for="r in rows" :key="r.name" class="ofbars__row">
          <span class="ofbars__name">{{ r.name }}</span>
          <span class="ofbars__bar" :style="{ width: (r.bytes / maxBytes) * 100 + '%', background: r.color }" />
        </div>
      </div>
      <p class="simcard__read">
        未融合 = 读a读b写t1 | 读t1读c写t2 | 读t2写t3 | 读t3写out = 8 趟；分块融合 = 读a读b读c写out = 4 趟（理论下限）。
        算术强度全部远小于 1 FLOP/Byte，说明这些算子<b>彻底 memory-bound</b>：
        <b>换更强的算力芯片一点用都没有，只有减少访存才有用</b>。
      </p>
    </div>

    <div class="simcard__block">
      <h4>本机实测：规模越大，分块融合收益越明显</h4>
      <p class="simcard__sub">工作集超过 cache 后，多走的每一趟内存都变成实打实的时间。纵轴对数。</p>
      <LineChart
        :series="scaleChart"
        x-label="数组规模 N"
        y-label="耗时 (ms，对数)"
        y-log
        :x-ticks="SCALE.map((s) => ({ v: s.n, label: s.n >= 1 ? s.n + 'M' : s.n * 1000 + 'K' }))"
        :y-format="(v: number) => v.toFixed(1)"
      />
    </div>

    <div class="simcard__block">
      <h4>Python 层写不出有效的融合：块大小的两难</h4>
      <p class="simcard__sub">
        N=8M，未融合基线 14.29ms。块要小到能进 cache，又要大到能摊薄每块 5 次 numpy 调用的解释器开销——
        两个要求直接冲突，最优出现在 64KB~256KB 的折中处，且始终追不上真正的融合 kernel。
      </p>
      <LineChart
        :series="blockChart"
        x-label="块工作集大小（KB，对数）"
        y-label="相对未融合的加速比"
        x-log
        :x-format="(v: number) => (v >= 1000 ? v / 1000 + 'MB' : v + 'KB')"
        :y-format="(v: number) => v.toFixed(2) + 'x'"
      />
      <p class="simcard__read">
        融合必须发生在编译产物内部——生成的机器码里是一个真正的融合循环，每个元素只付一次循环开销，
        而不是每块付 5 次解释器调用。这就是 TVM / TorchInductor 的核心价值。
      </p>
    </div>

    <p class="simcard__foot">
      访存模型是解析的；折线图与基线数字来自 <code>02_compile/demo2_operator_fusion.py</code>
      在本机的真实输出（未安装 torch，故无 torch.compile 对比）。
    </p>
  </div>
</template>

<style scoped>
.ofbars {
  display: grid;
  gap: 7px;
  margin-top: 12px;
}
.ofbars__row {
  display: grid;
  grid-template-columns: 170px 1fr;
  align-items: center;
  gap: 10px;
  font-size: 12.5px;
}
.ofbars__name {
  color: var(--vp-c-text-2);
}
.ofbars__bar {
  height: 11px;
  border-radius: 4px;
  min-width: 3px;
  transition: width 0.2s;
}
@media (max-width: 640px) {
  .ofbars__row {
    grid-template-columns: 120px 1fr;
    font-size: 12px;
  }
}
</style>
