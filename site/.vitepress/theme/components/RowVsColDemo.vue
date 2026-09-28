<script setup lang="ts">
/**
 * RowVsColDemo —— 01_data_storage/demo1 的交互式分析。
 * 训练只用一小部分字段时，行存为什么读不动。
 */
import { computed, ref } from 'vue'
import { fmtBytes } from '../lib/format'
import LineChart from './LineChart.vue'

const N_COLS = 40
// 本机实测（见真实输出）：zlib-6 压缩比
const ROW_CR = 5.55
const COL_CR = 6.87
const BW = 17.7e9 // 实测扫描带宽 B/s（64MB / 3.62ms）
const COLTYPE = [
  { name: '时间戳(单调)', cr: 253.57 },
  { name: '稀疏ID(长尾)', cr: 5.36 },
  { name: '低基数枚举', cr: 9.4 },
  { name: '稠密浮点', cr: 3.51 },
]

const nRows = ref(200_000)
const P = ref(3)
const N_LIST = [50_000, 200_000, 1_000_000, 5_000_000]

const rowBytes = computed(() => nRows.value * N_COLS * 8)
const colBytes = computed(() => nRows.value * P.value * 8)
const rowMs = computed(() => (rowBytes.value / BW) * 1e3)
const colMs = computed(() => (colBytes.value / BW) * 1e3)

const curve = computed(() => {
  const row = { name: '行存（扫全部 40 字段）', color: 'var(--sim-red)', points: [] as Array<[number, number]> }
  const col = { name: '列存（只扫投影列）', color: 'var(--sim-green)', points: [] as Array<[number, number]> }
  for (let p = 1; p <= N_COLS; p++) {
    row.points.push([p, nRows.value * N_COLS * 8])
    col.points.push([p, nRows.value * p * 8])
  }
  return [row, col]
})

const DAILY_ROWS = 1e11
const RAW_PB_PER_DAY = 100 // 1KB/条 × 1000 亿条
const diskRow = RAW_PB_PER_DAY / ROW_CR
const diskCol = RAW_PB_PER_DAY / COL_CR
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">01 / demo1 · 交互式分析</div>
        <h3 class="simcard__title">行存 vs 列存：训练只读 3/40 个字段，行存却要搬全部字节</h3>
      </div>
      <div class="simcard__badge">压缩比为 zlib-6 实测</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>样本行数 <b>{{ nRows.toLocaleString() }}</b></label>
        <div class="simcard__seg">
          <button v-for="n in N_LIST" :key="n" :class="{ 'is-on': nRows === n }" @click="nRows = n">
            {{ n >= 1e6 ? n / 1e6 + 'M' : n / 1e3 + 'K' }}
          </button>
        </div>
      </div>
      <div class="simcard__field">
        <label>训练投影的字段数 <b>{{ P }} / {{ N_COLS }}</b> <span class="simcard__hint">（一次实验通常只关心几十个特征里的一小部分）</span></label>
        <input v-model.number="P" type="range" min="1" max="40" step="1" />
      </div>
    </div>

    <div class="simcard__metrics">
      <div class="simcard__metric"><span>行存扫描</span><b>{{ fmtBytes(rowBytes) }}</b></div>
      <div class="simcard__metric"><span>列存扫描</span><b>{{ fmtBytes(colBytes) }}</b></div>
      <div class="simcard__metric is-hot"><span>I/O 降为</span><b>{{ ((P / N_COLS) * 100).toFixed(1) }}%</b></div>
      <div class="simcard__metric"><span>估算扫描耗时 行/列</span><b>{{ rowMs.toFixed(2) }} / {{ colMs.toFixed(2) }} ms</b></div>
    </div>

    <div class="simcard__block">
      <h4>扫描字节随投影字段数的变化</h4>
      <p class="simcard__sub">
        行存的字节量是水平线（必须整行搬进来再丢掉不要的字段）；列存随投影字段数线性下降。
      </p>
      <LineChart
        :series="curve"
        x-label="投影字段数"
        y-label="扫描字节"
        :x-format="(v: number) => String(Math.round(v))"
        :y-format="(v: number) => fmtBytes(v)"
      />
    </div>

    <div class="simcard__block">
      <h4>为什么列存顺带把压缩率打上去了</h4>
      <p class="simcard__sub">
        同一列的数据类型相同、取值域接近，熵更低。列存整表压缩 {{ COL_CR }}x、行存 {{ ROW_CR }}x。
        不同字段类型的可压缩性差异极大：
      </p>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead><tr><th>字段类型</th><th>压缩比</th></tr></thead>
          <tbody>
            <tr v-for="c in COLTYPE" :key="c.name"><td class="is-name">{{ c.name }}</td><td>{{ c.cr.toFixed(2) }}x</td></tr>
          </tbody>
        </table>
      </div>
      <p class="simcard__read">
        时间戳类列可压缩 250x 以上（单调、差分后近乎全零），稠密浮点只有 3.5x。
        列存把同类数据放一起，低基数枚举列还能进一步用字典/RLE 编码。
      </p>
    </div>

    <div class="simcard__block">
      <h4>外推到千亿级/日（单样本 1KB，1000 亿条/日 = 100 PB/日原始）</h4>
      <div class="simcard__metrics">
        <div class="simcard__metric"><span>行存落盘</span><b>{{ diskRow.toFixed(1) }} PB/日</b></div>
        <div class="simcard__metric"><span>列存落盘</span><b>{{ diskCol.toFixed(1) }} PB/日</b></div>
        <div class="simcard__metric is-hot"><span>列存一次实验扫描</span><b>{{ (diskCol * P / N_COLS).toFixed(2) }} PB</b></div>
      </div>
      <p class="simcard__read">
        列存的代价是写入要攒批（column chunk）：批越大压缩越好但数据越不新鲜，批越小则元数据爆炸。
        这正是<b>流式 Lakehouse</b>（Iceberg / Hudi / Paimon）要解决的核心矛盾。
      </p>
    </div>

    <p class="simcard__foot">
      压缩比与带宽来自 <code>01_data_storage/demo1_row_vs_col.py</code> 在本机的真实输出；
      扫描耗时用实测带宽线性外推。
    </p>
  </div>
</template>
