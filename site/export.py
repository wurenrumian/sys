#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把整个站点导出成**一个自包含的 HTML 文件**（离线可看、可分享）。

    python3 site/export.py                # 跑一遍全部 demo 并导出到 site/dist/index.html
    python3 site/export.py --skip-run     # 不跑 demo, 只打包讲解与源码

导出的快照里没有服务器, 所以不能调参重跑 —— 输出是导出时刻跑出来的结果。
要交互就用 `python3 site/serve.py`。两者共用同一份前端代码。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
DIST = os.path.join(HERE, "dist")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true", help="不实际运行 demo")
    ap.add_argument("--out", default=os.path.join(DIST, "index.html"))
    a = ap.parse_args()

    cat = json.loads(read(os.path.join(HERE, "catalog.json")))
    cat["env"] = {"python": sys.version.split()[0], "live": False}
    try:
        import numpy
        cat["env"]["numpy"] = numpy.__version__
    except ImportError:
        cat["env"]["numpy"] = None
    try:
        import torch
        cat["env"]["torch"] = torch.__version__
    except Exception:
        cat["env"]["torch"] = None

    docs, sources, outputs = {}, {}, {}
    for m in cat["modules"]:
        if m.get("doc"):
            p = os.path.join(ROOT, m["doc"])
            docs[m["doc"]] = read(p) if os.path.isfile(p) else "（缺少这份文档）"
        for d in m.get("demos", []):
            sources[d["file"]] = read(os.path.join(ROOT, d["file"]))
            if d.get("doc"):
                dp = os.path.join(ROOT, d["doc"])
                docs[d["doc"]] = read(dp) if os.path.isfile(dp) else "（缺少这份文档）"
            if a.skip_run:
                continue
            print(f"运行 {d['file']} ...", end="", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "-u", d["file"]], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=600)
            outputs[d["file"]] = r.stdout + (r.stderr or "")
            print(f" {time.time()-t0:.1f}s")

    payload = {"catalog": cat, "docs": docs, "sources": sources, "outputs": outputs}
    # 内联进 <script> 里的 JSON, 必须防止内容里出现 "</script>" 把标签提前闭合
    payload_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{cat.get('title', 'AI 基础设施')}（离线快照）</title>
<style>
{read(os.path.join(STATIC, 'style.css'))}
</style>
</head>
<body>
<div id="app">
  <nav id="side">
    <h1>AI 基础设施</h1>
    <p class="sub">讲解 · 演示 · 逐条验证（离线快照）</p>
    <div id="nav"></div>
    <div class="env" id="env"></div>
  </nav>
  <main id="main"><p class="lead">加载中…</p></main>
</div>
<script>window.__SNAPSHOT__ = {payload_js};</script>
<script>
{read(os.path.join(STATIC, 'md.js'))}
</script>
<script>
{read(os.path.join(STATIC, 'app.js'))}
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n已导出: {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print("直接用浏览器打开这个文件即可，无需任何服务。")


if __name__ == "__main__":
    main()
