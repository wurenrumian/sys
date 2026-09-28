<script setup lang="ts">
/**
 * PythonOverheadDemo —— 02_compile/demo1 的交互式分析。
 *
 * 纯 Python 与 numpy 的每元素成本是**本机实测**（见真实输出），
 * 这里让读者调节规模，看清"解释开销 → 内存带宽"这个瓶颈转移。
 */
import { computed, ref } from 'vue'
import { fmtMs } from '../lib/format'

interface Style {
  name: string
  ns: number
  note?: string
  trap?: boolean
}
// ns/元素 全部来自 02_compile_demo1 的真实运行输出
const STYLES: Style[] = [
  { name: '纯 Python 循环', ns: 38.8, note: '逐字节码 + 装箱拆箱' },
  { name: '列表推导 + sum', ns: 40.0, note: '仍是 PyObject 操作' },
  { name: 'numpy 向量化', ns: 3.6, note: '循环下沉到 C' },
  { name: 'numpy out=(零分配)', ns: 3.9, note: '分配不是瓶颈' },
  { name: 'numpy 布尔掩码', ns: 8.5, note: 'gather/scatter 陷阱', trap: true },
]
const BANDWIDTH_GBs = 7.7 // 实测有效带宽
const N_BYTES_NAIVE = 7 * 4 // 7 趟全尺寸 float32

const N = ref(2_000_000)
const N_LIST = [200_000, 1_000_000, 2_000_000, 10_000_000]

const rows = computed(() =>
  STYLES.map((s) => ({ ...s, ms: (s.ns * N.value) / 1e6 })),
)
const base = computed(() => rows.value[0].ms)

const memBytes = computed(() => N.value * N_BYTES_NAIVE)
const memMs = computed(() => memBytes.value / (BANDWIDTH_GBs * 1e9) * 1e3)
const intensity = computed(() => (N.value * 3) / memBytes.value)

const maxNs = computed(() => Math.max(...STYLES.map((s) => s.ns)))
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">02 / demo1 · 交互式分析</div>
        <h3 class="simcard__title">Python 解释器开销：瓶颈是怎么从「解释器」变成「内存带宽」的</h3>
      </div>
      <div class="simcard__badge">每元素成本为本机实测</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>元素个数 N <b>{{ N.toLocaleString() }}</b></label>
        <div class="simcard__seg">
          <button v-for="n in N_LIST" :key="n" :class="{ 'is-on': N === n }" @click="N = n">
            {{ n >= 1e6 ? n / 1e6 + 'M' : n / 1e3 + 'K' }}
          </button>
        </div>
        <input v-model.number="N" type="range" min="100000" max="10000000" step="100000" style="margin-top: 10px" />
      </div>
    </div>

    <div class="simcard__block">
      <h4>五种写法在 N = {{ N.toLocaleString() }} 下的估计耗时</h4>
      <p class="simcard__sub">每元素纳秒数取自真实运行；耗时 = ns/元素 × N。</p>
      <div class="pobars">
        <div v-for="r in rows" :key="r.name" class="pobars__row">
          <span class="pobars__name" :class="{ 'is-trap': r.trap }">{{ r.name }}</span>
          <span class="pobars__bar" :class="{ 'is-trap': r.trap }" :style="{ width: (r.ns / maxNs) * 100 + '%' }" />
          <span class="pobars__val">{{ fmtMs(r.ms, 1) }}</span>
          <span class="pobars__speed">{{ (base / r.ms).toFixed(0) }}x</span>
        </div>
      </div>
      <p class="simcard__read">
        向量化把逐元素的解释开销摊薄，一次分派处理整个数组，
        {{ (38.8 / 3.6).toFixed(0) }} 倍加速就是这么来的。但注意两个陷阱：
        <code>out=</code> 零分配版和布尔掩码版<b>实测都比朴素向量化更慢</b>
        （分别慢 {{ (((3.9 / 3.6) - 1) * 100).toFixed(0) }}% 和
        {{ (8.5 / 3.6).toFixed(1) }}x：每元素 3.9ns / 8.5ns vs 3.6ns）——
        靠人肉猜哪种写法更快，成功率并不高。
      </p>
    </div>

    <div class="simcard__block">
      <h4>向量化之后的新瓶颈：内存带宽</h4>
      <div class="simcard__metrics">
        <div class="simcard__metric"><span>搬运字节（7 趟）</span><b>{{ (memBytes / 1e6).toFixed(0) }} MB</b></div>
        <div class="simcard__metric"><span>带宽下限耗时</span><b>{{ fmtMs(memMs, 2) }}</b></div>
        <div class="simcard__metric"><span>算术量</span><b>{{ (N * 3 / 1e6).toFixed(0) }} MFLOP</b></div>
        <div class="simcard__metric is-hot"><span>算术强度</span><b>{{ intensity.toFixed(2) }} FLOP/Byte</b></div>
      </div>
      <p class="simcard__read">
        算术强度远小于 1 FLOP/Byte，意味着这段代码是<b>彻底 memory-bound</b>：
        换更强的算力芯片一点用都没有，唯一的出路是<b>减少访存次数</b>——也就是算子融合（见下一个 demo）。
        当前配置下，带宽下限 {{ fmtMs(memMs, 2) }} vs 朴素向量化耗时
        {{ fmtMs(rows[2].ms, 2) }}，两者已经同一量级，说明瓶颈确实转移到了内存。
      </p>
    </div>

    <p class="simcard__foot">
      这是**模型**：常数（ns/元素、7.7 GB/s）来自 <code>02_compile/demo1_python_overhead.py</code>
      在本机的真实输出，模型本身是线性的，不含 cache 效应等二阶项。
    </p>
  </div>
</template>

<style scoped>
.pobars {
  display: grid;
  gap: 9px;
}
.pobars__row {
  display: grid;
  grid-template-columns: 150px 1fr 84px 48px;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.pobars__name {
  color: var(--vp-c-text-2);
}
.pobars__name.is-trap {
  color: var(--sim-red);
}
.pobars__bar {
  height: 12px;
  border-radius: 4px;
  background: var(--sim-blue);
  min-width: 3px;
  transition: width 0.2s;
}
.pobars__bar.is-trap {
  background: var(--sim-red);
}
.pobars__val {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.pobars__speed {
  text-align: right;
  color: var(--vp-c-text-3);
  font-variant-numeric: tabular-nums;
}
@media (max-width: 640px) {
  .pobars__row {
    grid-template-columns: 108px 1fr 70px 40px;
    font-size: 12px;
  }
}
</style>
