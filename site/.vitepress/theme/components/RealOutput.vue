<script setup lang="ts">
/**
 * RealOutput —— 回放某个 demo 在本机实跑得到的 stdout。
 * 用法：<RealOutput slug="01_data_storage_demo2_cache_tier"
 *                    label="01 / demo2 · cache_tier 真实输出"
 *                    command="python3 01_data_storage/demo2_cache_tier.py" />
 */
import DemoTerminal from './DemoTerminal.vue'

import t01d1 from '../data/01_data_storage_demo1_row_vs_col.txt?raw'
import t01d2 from '../data/01_data_storage_demo2_cache_tier.txt?raw'
import t01d3 from '../data/01_data_storage_demo3_data_quality.txt?raw'
import t02d1 from '../data/02_compile_demo1_python_overhead.txt?raw'
import t02d2 from '../data/02_compile_demo2_operator_fusion.txt?raw'
import t02d3 from '../data/02_compile_demo3_gpu_mem_frag.txt?raw'
import t02d4 from '../data/02_compile_demo4_dynamic_batch.txt?raw'
import t03d1 from '../data/03_network_demo1_traffic_conflict.txt?raw'
import t03d2 from '../data/03_network_demo2_allreduce.txt?raw'
import t03d3 from '../data/03_network_demo3_congestion.txt?raw'

const MAP: Record<string, string> = {
  '01_data_storage_demo1_row_vs_col': t01d1,
  '01_data_storage_demo2_cache_tier': t01d2,
  '01_data_storage_demo3_data_quality': t01d3,
  '02_compile_demo1_python_overhead': t02d1,
  '02_compile_demo2_operator_fusion': t02d2,
  '02_compile_demo3_gpu_mem_frag': t02d3,
  '02_compile_demo4_dynamic_batch': t02d4,
  '03_network_demo1_traffic_conflict': t03d1,
  '03_network_demo2_allreduce': t03d2,
  '03_network_demo3_congestion': t03d3,
}

defineProps<{ slug: string; label?: string; command?: string }>()
</script>

<template>
  <DemoTerminal
    v-if="MAP[slug]"
    :text="MAP[slug]"
    :title="label || slug"
    :command="command || ''"
  />
</template>
