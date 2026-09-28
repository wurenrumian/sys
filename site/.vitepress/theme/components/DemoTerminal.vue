<script setup lang="ts">
/**
 * DemoTerminal —— 回放脚本的【真实】运行输出。
 *
 * 用法：把脚本跑一遍，把 stdout 作为 text 传进来。
 * 刻意做成**手动运行**：进入页面不会自动播放，读者点「运行脚本」才开始逐行回放。
 * 面板高度固定，回放过程中不会把页面顶来顶去。
 */
import { computed, onBeforeUnmount, ref } from 'vue'

const props = withDefaults(
  defineProps<{
    text: string
    title?: string
    command?: string
    /** 每秒播放的行数 */
    linesPerSecond?: number
  }>(),
  { title: '真实运行输出', command: '', linesPerSecond: 26 },
)

const lines = computed(() => props.text.replace(/\s+$/, '').split('\n'))
const total = computed(() => lines.value.length)

const started = ref(false)
const shown = ref(0)
const playing = ref(false)
const speed = ref(1)
const pre = ref<HTMLElement | null>(null)

let raf = 0
let last = 0

function tick(ts: number) {
  if (!playing.value) return
  if (!last) last = ts
  const dt = (ts - last) / 1000
  last = ts
  const add = props.linesPerSecond * speed.value * dt
  if (add >= 1 || Math.random() < add) {
    shown.value = Math.min(total.value, shown.value + Math.max(1, Math.floor(add)))
    if (pre.value) pre.value.scrollTop = pre.value.scrollHeight
  }
  if (shown.value >= total.value) {
    playing.value = false
    return
  }
  raf = requestAnimationFrame(tick)
}

function play() {
  if (shown.value >= total.value) shown.value = 0
  playing.value = true
  last = 0
  cancelAnimationFrame(raf)
  raf = requestAnimationFrame(tick)
}
function pause() {
  playing.value = false
  cancelAnimationFrame(raf)
}
function toggle() {
  playing.value ? pause() : play()
}
/** 手动运行入口 */
function run() {
  started.value = true
  shown.value = 0
  play()
}
function replay() {
  shown.value = 0
  play()
}
function showAll() {
  pause()
  started.value = true
  shown.value = total.value
}

function lineClass(line: string) {
  const s = line.trim()
  if (s.startsWith('====') || s.startsWith('----')) return 'dim'
  if (/->|改善|反直觉|结论/.test(line)) return 'accent'
  if (/(\d|%|us|ms|Gbps)/.test(line) && /^\s/.test(line)) return 'data'
  return ''
}

onBeforeUnmount(() => cancelAnimationFrame(raf))
</script>

<template>
  <div class="demo-term">
    <div class="demo-term__bar">
      <span class="demo-term__dots"><i /><i /><i /></span>
      <span class="demo-term__title">{{ title }}</span>
      <span class="demo-term__spacer" />
      <template v-if="started">
        <button class="demo-term__btn" :title="playing ? '暂停' : '继续'" @click="toggle">
          {{ playing ? '暂停' : '继续' }}
        </button>
        <button class="demo-term__btn" title="从头再放一遍" @click="replay">重播</button>
        <button class="demo-term__btn" title="直接显示全部" @click="showAll">秒开</button>
        <span class="demo-term__speed">
          <button
            v-for="s in [1, 2, 4]"
            :key="s"
            class="demo-term__btn demo-term__btn--s"
            :class="{ 'is-on': speed === s }"
            @click="speed = s"
          >
            {{ s }}x
          </button>
        </span>
      </template>
      <button v-else class="demo-term__run" title="运行脚本并回放输出" @click="run">▶ 运行脚本</button>
    </div>

    <pre v-if="started" ref="pre" class="demo-term__pane demo-term__body"><code><span
      v-for="(l, i) in lines.slice(0, Math.max(shown, 1))"
      :key="i"
      class="demo-term__line"
      :class="lineClass(l)"
    >{{ l }}
</span></code></pre>

    <div v-else class="demo-term__pane demo-term__idle">
      <div class="demo-term__idle-icon">$</div>
      <div class="demo-term__idle-cmd">{{ command || 'python3 xxx.py' }}</div>
      <div class="demo-term__idle-hint">这一步需要手动运行——点右上角「▶ 运行脚本」，即回放它在作者本机的真实输出。</div>
    </div>

    <div class="demo-term__foot">
      <span v-if="started">{{ Math.min(shown, total) }} / {{ total }} 行</span>
      <span v-else>共 {{ total }} 行输出 · 未运行</span>
      <span v-if="started && shown >= total && total > 0" class="demo-term__done">✓ 回放完毕</span>
    </div>
  </div>
</template>

<style scoped>
.demo-term {
  margin: 18px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  overflow: hidden;
  background: var(--vp-code-block-bg, #1e1e20);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
}
.demo-term__bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  background: color-mix(in srgb, var(--vp-code-block-bg, #1e1e20) 80%, #fff 6%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 12px;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}
.demo-term__dots {
  display: inline-flex;
  gap: 5px;
  flex: none;
}
.demo-term__dots i {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #ff5f57;
}
.demo-term__dots i:nth-child(2) {
  background: #febc2e;
}
.demo-term__dots i:nth-child(3) {
  background: #28c840;
}
.demo-term__title {
  font-weight: 600;
  color: var(--vp-c-text-1);
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.demo-term__spacer {
  flex: 1 1 auto;
}
.demo-term__btn {
  font: inherit;
  flex: none;
  white-space: nowrap;
  color: var(--vp-c-text-2);
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 2px 8px;
  cursor: pointer;
  line-height: 1.5;
}
.demo-term__btn:hover {
  color: var(--vp-c-text-1);
  background: rgba(255, 255, 255, 0.12);
}
.demo-term__run {
  font: inherit;
  flex: none;
  white-space: nowrap;
  font-weight: 600;
  color: #0b1020;
  background: var(--vp-c-brand-1);
  border: none;
  border-radius: 6px;
  padding: 3px 12px;
  cursor: pointer;
}
.demo-term__run:hover {
  filter: brightness(1.08);
}
.demo-term__speed {
  display: inline-flex;
  gap: 3px;
}
.demo-term__btn--s.is-on {
  color: #0b1020;
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}
/* 固定面板高度：回放不会撑高页面 */
.demo-term__pane {
  height: clamp(240px, 52vh, 440px);
  margin: 0;
}
.demo-term__body {
  padding: 12px 14px;
  overflow: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 12.5px;
  line-height: 1.55;
  color: #d4d4d8;
  background: transparent;
}
.demo-term__body code {
  background: transparent;
  padding: 0;
  font-size: inherit;
}
.demo-term__line {
  white-space: pre;
  display: block;
  min-height: 1.55em;
}
.demo-term__line.dim {
  color: #6b7280;
}
.demo-term__line.accent {
  color: #fbbf24;
}
.demo-term__line.data {
  color: #93c5fd;
}
.demo-term__idle {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px;
  text-align: center;
}
.demo-term__idle-icon {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 18px;
  color: #7dd3fc;
  background: rgba(125, 211, 252, 0.12);
  border: 1px solid rgba(125, 211, 252, 0.3);
}
.demo-term__idle-cmd {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  color: #7dd3fc;
}
.demo-term__idle-hint {
  font-size: 12px;
  color: var(--vp-c-text-3);
  max-width: 420px;
  line-height: 1.7;
}
.demo-term__foot {
  display: flex;
  justify-content: space-between;
  padding: 5px 12px;
  font-size: 11px;
  color: var(--vp-c-text-3);
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.demo-term__done {
  color: #4ade80;
}
</style>
