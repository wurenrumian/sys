<script setup lang="ts">
/**
 * DataQualityDemo —— 01_data_storage/demo3 的浏览器版。
 * 模型代码完全不变，只让上游数据出问题：AUC 会掉，而 PSI 常常抓不到。
 */
import { computed, onMounted, ref } from 'vue'
import {
  INCIDENTS,
  injectIncident,
  makeData,
  maxPsi,
  auc,
  column,
  psi,
  score,
  type Dataset,
} from '../lib/dq'
import LineChart from './LineChart.vue'

const N_TRAIN = 15_000
const N_SERVE = 8_000

const seed = ref(0)
const train = ref<Dataset | null>(null)
const serve = ref<Dataset | null>(null)
const baseAuc = ref(0)

function gen() {
  train.value = makeData(1 + seed.value, N_TRAIN)
  serve.value = makeData(2 + seed.value, N_SERVE)
  baseAuc.value = auc(serve.value.y, score(serve.value.X, serve.value.n))
}

interface Row {
  id: string
  label: string
  auc: number
  d: number
  psi: number
  feat: number
  flag: string
}
const rows = computed<Row[]>(() => {
  const tr = train.value
  const se = serve.value
  if (!tr || !se) return []
  return INCIDENTS.map((inc) => {
    const bad = injectIncident(inc.id, se.X, se.n, 2 + seed.value)
    const a = auc(se.y, score(bad, se.n))
    const mp = maxPsi(tr.X, tr.n, bad, se.n)
    const flag = mp.value > 0.25 ? '报警' : mp.value > 0.1 ? '关注' : '正常'
    return { id: inc.id, label: inc.label, auc: a, d: a - baseAuc.value, psi: mp.value, feat: mp.feat, flag }
  })
})

// 敏感度：feat_2 被随机值替换（PSI 盲区）
const sensitivity = computed(() => {
  const tr = train.value
  const se = serve.value
  if (!tr || !se) return { dAuc: [], psi: [] }
  const ratios = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
  const pts = ratios.map((r) => {
    const bad = injectIncident('skew', se.X, se.n, 2 + seed.value, r)
    const a = auc(se.y, score(bad, se.n))
    const p = psi(column(tr.X, tr.n, 2), column(bad, se.n, 2))
    return { r, d: (a - baseAuc.value) * 100, p }
  })
  return {
    dAuc: [{ name: 'ΔAUC（百分点）', color: 'var(--sim-red)', points: pts.map((x) => [x.r * 100, x.d] as [number, number]) }],
    psi: [{ name: 'feat_2 的 PSI', color: 'var(--sim-blue)', points: pts.map((x) => [x.r * 100, x.p] as [number, number]) }],
  }
})

onMounted(gen)
</script>

<template>
  <div class="simcard">
    <div class="simcard__head">
      <div>
        <div class="simcard__eyebrow">01 / demo3 · 实时模拟</div>
        <h3 class="simcard__title">DCAI：模型一行没改，上游数据出问题，线上 AUC 就会掉</h3>
      </div>
      <div class="simcard__badge">模型权重取自真实训练</div>
    </div>

    <div class="simcard__controls">
      <div class="simcard__row">
        <div class="simcard__field">
          <label>健康状态下的线上 AUC <b>{{ baseAuc.toFixed(4) }}</b> <span class="simcard__hint">（训练 {{ N_TRAIN.toLocaleString() }} / 线上 {{ N_SERVE.toLocaleString() }}，12 维特征）</span></label>
        </div>
        <button class="simcard__btn" @click="seed++; gen()">换一组数据 ↻</button>
      </div>
    </div>

    <div class="simcard__block" style="border-top: none; margin-top: 16px; padding-top: 0">
      <h4>五种情形：AUC 的实际掉幅 vs PSI 的读数</h4>
      <div class="simcard__tablewrap">
        <table class="simcard__table">
          <thead>
            <tr><th>情形</th><th>AUC</th><th>ΔAUC</th><th>最大 PSI</th><th>责任特征</th><th>判定</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in rows" :key="r.id">
              <td class="is-name">{{ r.label }}</td>
              <td>{{ r.auc.toFixed(4) }}</td>
              <td :class="{ 'is-hot': r.d < -0.01 }">{{ r.d >= 0 ? '+' : '' }}{{ r.d.toFixed(4) }}</td>
              <td>{{ r.psi.toFixed(3) }}</td>
              <td>feat_{{ r.feat }}</td>
              <td>
                <span :class="['dqflag', r.flag === '正常' ? 'dqflag--ok' : r.flag === '关注' ? 'dqflag--warn' : 'dqflag--bad']">{{ r.flag }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="simcard__read">
        最该注意的不是 PSI 抓到了什么，而是它<b>漏掉了什么</b>：
        「离在线不一致」和「特征不新鲜」两行的 AUC 已经掉了 2 个点以上，PSI 却判定「正常」。
        因为 PSI 是<b>单特征边缘分布</b>指标，对「分布没变但语义错位」这类事故天然盲视——
        而这恰恰是推荐系统最高频的事故类型。
      </p>
    </div>

    <div class="simcard__block">
      <h4>为什么 PSI 是盲的：污染比例 → PSI → AUC</h4>
      <p class="simcard__sub">
        以 feat_2 被同分布随机值替换为例（离在线不一致）：污染比例从 0% 到 100%，
        AUC 单调下降，而 PSI 全程约等于 0。
      </p>
      <div class="dqgrid">
        <div>
          <div class="dqgrid__cap">AUC 实际掉幅</div>
          <LineChart :series="sensitivity.dAuc" x-label="污染比例 (%)" y-label="ΔAUC（百分点）" :x-format="(v: number) => v.toFixed(0) + '%'" :y-format="(v: number) => v.toFixed(1)" />
        </div>
        <div>
          <div class="dqgrid__cap">PSI 读数</div>
          <LineChart :series="sensitivity.psi" x-label="污染比例 (%)" y-label="PSI" :x-format="(v: number) => v.toFixed(0) + '%'" :y-format="(v: number) => v.toFixed(3)" />
        </div>
      </div>
      <p class="simcard__read">
        结论是<b>单一指标兜不住</b>，工业上必须分层组合：统计类（空值率/均值）抓任务挂掉、
        分布类（PSI/KS）抓口径变更、一致性对拍抓训推不一致、特征-label 关联漂移抓语义错位、
        影子流量 AUC 兜底。真正要建的是能把「数据异常量」映射到「业务指标掉幅」的<b>可量化体系</b>。
      </p>
    </div>

    <p class="simcard__foot">
      逻辑回归权重由 <code>01_data_storage/demo3_data_quality.py</code> 的训练流程在本机训练得到并内嵌；
      数据在浏览器端用同分布重新生成，故数值与脚本有统计波动（基线 AUC 脚本为 0.8743）。
    </p>
  </div>
</template>

<style scoped>
.dqgrid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.dqgrid__cap {
  font-size: 12.5px;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
@media (max-width: 760px) {
  .dqgrid {
    grid-template-columns: 1fr;
  }
}
.dqflag {
  font-size: 11.5px;
  padding: 1px 7px;
  border-radius: 999px;
}
.dqflag--ok {
  color: var(--sim-green);
  background: color-mix(in srgb, var(--sim-green) 14%, transparent);
}
.dqflag--warn {
  color: var(--sim-amber);
  background: color-mix(in srgb, var(--sim-amber) 16%, transparent);
}
.dqflag--bad {
  color: var(--sim-red);
  background: color-mix(in srgb, var(--sim-red) 14%, transparent);
}
</style>
