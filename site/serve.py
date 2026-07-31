#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地站点服务器 —— 让这些 demo 可以在浏览器里读、调参、真跑、验证。

    python3 site/serve.py            # 然后打开 http://127.0.0.1:8777
    python3 site/serve.py --port 9000 --open

只用 Python 标准库, 没有任何依赖。只监听 127.0.0.1, 不对外暴露。

设计要点:
  * 页面里点"运行"是**真的 fork 一个 python 子进程**去跑那个 demo, 输出流式回传,
    不是预先录好的快照 —— 因为这个项目的价值就在于结论可复现、可推翻。
  * 表单里的参数只会作为**环境变量**传给子进程(见 common/params.py), 不改任何代码。
  * 只允许运行 catalog.json 里登记过的文件、只接受 catalog.json 里声明过的参数名,
    参数值还要过一遍正则 —— 本地工具也不该给自己留一个命令注入的口子。
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
CATALOG_PATH = os.path.join(HERE, "catalog.json")

SAFE_VALUE = re.compile(r"^[A-Za-z0-9._+-]{1,32}$")
RUN_TIMEOUT_S = 300

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml"}


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def catalog_index(cat):
    """demo 文件路径 -> demo 定义, 用于白名单校验。"""
    out = {}
    for m in cat["modules"]:
        for d in m.get("demos", []):
            out[d["file"]] = d
    return out


def doc_whitelist(cat):
    docs = set()
    for m in cat["modules"]:
        if m.get("doc"):
            docs.add(m["doc"])
        for extra in m.get("extra_docs", []):
            docs.add(extra["path"])
    return docs


class Handler(BaseHTTPRequestHandler):
    server_version = "sys-demo-site"

    # ------------------------------------------------------------ 基础工具
    def log_message(self, fmt, *args):
        if "/api/run" in (args[0] if args else ""):
            sys.stderr.write("  运行: %s\n" % args[0])

    def _send(self, code, body, ctype="text/plain; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def _static(self, name):
        path = os.path.normpath(os.path.join(STATIC, name))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            return self._send(404, "not found")
        with open(path, "rb") as f:
            body = f.read()
        self._send(200, body, MIME.get(os.path.splitext(path)[1], "application/octet-stream"))

    # ------------------------------------------------------------ 路由
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path

        if p in ("/", "/index.html"):
            return self._static("index.html")
        if p.startswith("/static/"):
            return self._static(p[len("/static/"):])
        if p == "/api/catalog":
            return self.api_catalog()
        if p == "/api/doc":
            return self.api_doc(q)
        if p == "/api/source":
            return self.api_source(q)
        if p == "/api/run":
            return self.api_run(q)
        return self._send(404, "not found")

    # ------------------------------------------------------------ API
    def api_catalog(self):
        cat = load_catalog()
        try:
            import numpy
            npv = numpy.__version__
        except ImportError:
            npv = None
        try:
            import torch
            tv = torch.__version__
        except Exception:
            tv = None
        cat["env"] = {
            "python": sys.version.split()[0],
            "numpy": npv,
            "torch": tv,
            "root": ROOT,
            "live": True,
        }
        self._json(cat)

    def api_doc(self, q):
        path = (q.get("path") or [""])[0]
        if path not in doc_whitelist(load_catalog()):
            return self._send(403, "not allowed")
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            return self._send(404, "missing: " + path)
        with open(full, encoding="utf-8") as f:
            self._send(200, f.read(), "text/markdown; charset=utf-8")

    def api_source(self, q):
        path = (q.get("path") or [""])[0]
        if path not in catalog_index(load_catalog()):
            return self._send(403, "not allowed")
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            self._send(200, f.read())

    def api_run(self, q):
        cat = load_catalog()
        idx = catalog_index(cat)
        path = (q.get("path") or [""])[0]
        demo = idx.get(path)
        if not demo:
            return self._send(403, "not allowed")

        # 只接受这个 demo 声明过的参数, 值还要过一遍白名单正则
        allowed = {p["name"] for p in demo.get("params", [])}
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        used = {}
        for k, vs in q.items():
            if k == "path" or not vs:
                continue
            if k not in allowed:
                return self._send(400, f"未知参数: {k}")
            v = vs[0].strip()
            if v == "":
                continue
            if not SAFE_VALUE.match(v):
                return self._send(400, f"参数 {k} 的值不合法")
            env[k] = v
            used[k] = v

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def sse(event, data):
            payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        proc = subprocess.Popen(
            [sys.executable, "-u", path], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True,
        )
        timed_out = {"v": False}

        def killer():
            timed_out["v"] = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

        timer = threading.Timer(RUN_TIMEOUT_S, killer)
        timer.start()
        try:
            sse("start", {"path": path, "params": used})
            for line in proc.stdout:
                sse("line", line.rstrip("\n"))
            code = proc.wait()
            sse("done", {"code": code, "timeout": timed_out["v"]})
        except (BrokenPipeError, ConnectionResetError):
            # 浏览器关掉了这次运行 —— 顺手把子进程也杀掉, 别留孤儿
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        finally:
            timer.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    a = ap.parse_args()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"站点已启动: {url}")
    print(f"  项目根目录: {ROOT}")
    print("  Ctrl-C 退出")
    if a.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
