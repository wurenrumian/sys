import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI 基础设施系统实验室',
  description: '数据存储、编译优化与网络系统的最小实验',
  cleanUrls: true,
  themeConfig: {
    nav: [
      { text: '总览', link: '/' },
      { text: '数据存储', link: '/01_data_storage/' },
      { text: '编译方向', link: '/02_compile/' },
      { text: '网络方向', link: '/03_network/' },
    ],
    sidebar: [
      {
        text: 'AI 基础设施方向',
        items: [
          { text: '总览', link: '/' },
          { text: '数据计算和存储中台', link: '/01_data_storage/' },
          { text: '基础库团队 · 编译方向', link: '/02_compile/' },
          { text: '基础库团队 · 网络方向', link: '/03_network/' },
        ],
      },
    ],
    search: { provider: 'local' },
    outline: [2, 3],
    docFooter: { prev: '上一篇', next: '下一篇' },
  },
})
