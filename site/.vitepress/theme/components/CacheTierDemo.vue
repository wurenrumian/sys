<script setup lang="ts">
/**
 * CacheTierDemo —— 01_data_storage/demo2 的浏览器版。
 * 长尾访问下，缓存值多少钱；以及"提命中率"和"让某层更快"哪个更值。
 */
import { computed, onMounted, ref, watch } from 'vue'
import {
  N_KEYS,
  TIERS,
  TIER_DIST,
  effectiveLatency,
  genTrace,
  hitRate,
  tierCost,
  type Trace,
} from '../lib/cache'
import { fmtInt, fmtUs } from '../lib/format'
import LineChart from './LineChart.vue'

const isClient = typeof window !== 'undefined'
const N_REQ = 300_000

const zipfA = ref(1.15)
const churn = ref(false)
const capPct = ref(1.0) // 缓存容量占比 %
const policy = ref<'lru' | 'lfu'>('lru')
const penetration = ref(0.02) // 穿透到对象存储的比例

const busy = ref(false)
const trace = ref<Trace | null>(null)

let timer: ReturnType<typeof setTimeout> | undefined
function regen() {
  busy.value = true
  clearTimeout(timer)
  timer = setTimeout(() => {
    trace.value = genTrace({ nReq: N_REQ, a: zipfA.value, churnEvery: churn.value ? 50_000 : null })
    busy.value = false
  }, 40)
}
if (isClient) watch([zipfA, churn], regen)

const CAPS = [0.001, 0.005, 0.01, 0.05, 0.1]
const curve = computed(() => {
  const tr = trace.value
  if (!tr) return []
  const rows = CAPS.map((r) => {
    const cap = Math.max(1, Math.floor(N_KEYS * r))
    return { r, lru: hitRate(tr, 'lru', cap), lfu: hitRate(tr, 'lfu', cap) }
  })
  return [
    { name: 'LRU', color: 'var(--sim-blue)', points: rows.map((x) => [x.r * 100, x.lru * 100] as [number, number]) },
    { name: 'LFU', color: 'var(--sim-violet)', points: rows.map((x) => [x.r * 100, x.lfu * 100] as [number, number]) },
  ]
})

const currentHit = computed(() => {
  const tr = trace.value
  if (!tr) return 0
  const cap = Math.max(1, Math.floor(N_KEYS * (capPct.value / 100)))
  return hitRate(tr, policy.value, cap)
})

// ---------------------------------------------------------------- 有效延迟
const base3 = computed(() => Math.max(0, 1 - 0.3 - 0.4 - 0.2 - penetration.value))
const baseHit = computed(() => [0.3, 0.4, 0.2, base3.value])
const l0 = computed(() => effectiveLatency(baseHit.value))
const variant = computed(() => {
  const p = penetration.value
  const fasterTiers = TIERS.map((t, i) => (i === 3 ? { ...t, lat: t.lat / 10 } : t))
  return {
    hbm: effectiveLatency([0.31, 0.39, 0.2, base3.value]),
    nvme: effectiveLatency(baseHit.value, fasterTiers),
    leak: effectiveLatency([0.3, 0.4, 0.2, base3.value + p / 2]),
  }
})
const latencyCurve = computed(() => {
  const pts: Array<[number, number]> = []
  for (let e = -4; e <= -1.3; e += 0.15) {
    const p = Math.pow(10, e)
    const b3 = Math.max(0, 1 - 0.3 - 0.4 - 0.2 - p)
    pts.push([p * 100, effectiveLatency([0.3, 0.4, 0.2, b3])])
  }
  return [{ name: '有效延迟', color: 'var(--sim-amber)', points: pts }]
})

const cost = computed(() => tierCost(TIER_DIST))
const costTotal = cost.value
const hbmCost = TIERS[0].cost

onMounted(regen)
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">01 / demo2 · 实时模拟</div>
        <h3 class="simcard__title">多级存储 + Embedding Cache：长尾分布下缓存值多少钱</h3>
      </div>
      <div class="simcard__badge" :class="{ 'is-busy': busy }">{{ busy ? '计算中…' : '与脚本一致' }}</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__field">
        <label>访问分布的集中度 Zipf α <b>{{ zipfA.toFixed(2) }}</b></label>
        <input v-model.number="zipfA" type="range" min="0.8" max="1.5" step="0.01" />
      </div>
      <div class="simcard__row">
        <div class="simcard__field">
          <label>缓存容量 <b>{{ capPct.toFixed(2) }}%</b>（全表 {{ fmtInt(N_KEYS) }} key）</label>
          <input v-model.number="capPct" type="range" min="0.05" max="10" step="0.05" />
        </div>
        <div class="simcard__field">
          <label>策略</label>
          <div class="simcard__seg">
            <button :class="{ 'is-on': policy === 'lru' }" @click="policy = 'lru'">LRU</button>
            <button :class="{ 'is-on': policy === 'lfu' }" @click="policy = 'lfu'">LFU</button>
          </div>
        </div>
        <div class="simcard__field">
          <label>热点 churn</label>
          <div class="simcard__seg">
            <button :class="{ 'is-on': !churn }" @click="churn = false">稳态</button>
            <button :class="{ 'is-on': churn }" @click="churn = true">每 5 万请求换热点</button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="trace" class="simcard__metrics">
      <div class="simcard__metric"><span>访问过的 key</span><b>{{ fmtInt(trace.distinct) }}</b></div>
      <div class="simcard__metric"><span>最热 1% 承接</span><b>{{ (trace.top1pct * 100).toFixed(1) }}%</b></div>
      <div class="simcard__metric is-hot"><span>{{ policy.toUpperCase() }} 命中率</span><b>{{ (currentHit * 100).toFixed(1) }}%</b></div>
      <div class="simcard__metric"><span>容量条目数</span><b>{{ fmtInt(N_KEYS * capPct / 100) }}</b></div>
    </div>

    <div class="simcard__block">
      <h4>命中率对容量是强凹的</h4>
      <p class="simcard__sub">
        前 1% 的容量就买到了绝大部分收益，之后急剧衰减——这是缓存在推荐场景性价比极高的根本原因。
      </p>
      <LineChart :series="curve" x-label="缓存容量占比 (%)" y-label="命中率 (%)" :x-format="(v: number) => v.toFixed(1)" :y-format="(v: number) => v.toFixed(0) + '%'" />
      <p class="simcard__read">
        稳态下 LFU 明显更强（记得住长期热点）；打开 churn 后 LFU 的优势被吃掉甚至反转，
        因为陈旧的高频次计数会挡住新热点进入缓存（cache pollution）。工业方案 W-TinyLFU
        用小 LRU 窗口吸收突发 + 主体 TinyLFU 保稳态 + 计数老化。
      </p>
    </div>

    <div class="simcard__block">
      <h4>「提命中率」和「让某层更快」哪个更值</h4>
      <div class="simcard__field" style="max-width: 420px">
        <label>穿透到对象存储的比例 <b>{{ (penetration * 100).toFixed(2) }}%</b></label>
        <input v-model.number="penetration" type="range" min="0.0005" max="0.05" step="0.0005" />
      </div>
      <div class="simcard__metrics">
        <div class="simcard__metric"><span>基线有效延迟</span><b>{{ fmtUs(l0) }}</b></div>
        <div class="simcard__metric"><span>HBM +1pt</span><b>改善 {{ (((l0 - variant.hbm) / l0) * 100).toFixed(1) }}%</b></div>
        <div class="simcard__metric"><span>NVMe 提速 10x</span><b>改善 {{ (((l0 - variant.nvme) / l0) * 100).toFixed(1) }}%</b></div>
        <div class="simcard__metric is-hot"><span>穿透率减半</span><b>改善 {{ (((l0 - variant.leak) / l0) * 100).toFixed(1) }}%</b></div>
      </div>
      <LineChart :series="latencyCurve" x-label="穿透率 (%)" y-label="有效延迟" x-log :x-format="(v: number) => v + '%'" :y-format="(v: number) => fmtUs(v)" />
      <p class="simcard__read">
        穿透率虽小，却可能贡献绝大部分平均延迟。此时唯一有效的动作是压穿透率，上层调优和硬件升级基本是噪声；
        等穿透率压到万分之几，压穿透的边际收益迅速衰减，上层的命中率分布才开始成为主要矛盾。
        <b>优化顺序是被这条延迟-穿透率曲线决定的，不是拍脑袋定的。</b>
      </p>
    </div>

    <div class="simcard__block">
      <h4>分层存储的代价</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead><tr><th>层级</th><th>延迟</th><th>容量占比</th><th>相对成本</th></tr></thead>
          <tbody>
            <tr v-for="(t, i) in TIERS" :key="t.name">
              <td class="is-name">{{ t.name }}</td>
              <td>{{ fmtUs(t.lat) }}</td>
              <td>{{ (TIER_DIST[i] * 100).toFixed(1) }}%</td>
              <td>{{ (TIER_DIST[i] * t.cost).toFixed(3) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="simcard__read">
        分层总成本是全 HBM 的 {{ ((costTotal / hbmCost) * 100).toFixed(1) }}%（降到 1/{{ Math.round(hbmCost / costTotal) }}），
        代价是延迟上升。分层存储的全部设计工作，就是在这条兑换曲线上找业务能接受的那个点。
      </p>
    </div>

    <p class="simcard__foot">
      模型与 <code>01_data_storage/demo2_cache_tier.py</code> 一致（Zipf 采样已按 numpy 的无界尾部校准，
      最热 1% 占比与脚本一致）；规模从 40 万请求降到 {{ fmtInt(N_REQ) }} 以保证交互流畅。
    </p>
  </div>
</template>
