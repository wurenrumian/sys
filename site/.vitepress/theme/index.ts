import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'

import DemoTerminal from './components/DemoTerminal.vue'
import RealOutput from './components/RealOutput.vue'
import LineChart from './components/LineChart.vue'
import TrafficConflictDemo from './components/TrafficConflictDemo.vue'
import AllReduceDemo from './components/AllReduceDemo.vue'
import CongestionDemo from './components/CongestionDemo.vue'
import RowVsColDemo from './components/RowVsColDemo.vue'
import CacheTierDemo from './components/CacheTierDemo.vue'
import DataQualityDemo from './components/DataQualityDemo.vue'
import PythonOverheadDemo from './components/PythonOverheadDemo.vue'
import OperatorFusionDemo from './components/OperatorFusionDemo.vue'
import GpuMemFragDemo from './components/GpuMemFragDemo.vue'
import DynamicBatchDemo from './components/DynamicBatchDemo.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('DemoTerminal', DemoTerminal)
    app.component('RealOutput', RealOutput)
    app.component('LineChart', LineChart)
    app.component('TrafficConflictDemo', TrafficConflictDemo)
    app.component('AllReduceDemo', AllReduceDemo)
    app.component('CongestionDemo', CongestionDemo)
    app.component('RowVsColDemo', RowVsColDemo)
    app.component('CacheTierDemo', CacheTierDemo)
    app.component('DataQualityDemo', DataQualityDemo)
    app.component('PythonOverheadDemo', PythonOverheadDemo)
    app.component('OperatorFusionDemo', OperatorFusionDemo)
    app.component('GpuMemFragDemo', GpuMemFragDemo)
    app.component('DynamicBatchDemo', DynamicBatchDemo)
  },
} satisfies Theme
