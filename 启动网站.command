#!/usr/bin/env bash
# 双击本文件即可启动网站（macOS Finder 里双击就行，不用敲命令）。
# 关掉这个终端窗口就等于关掉网站。
cd "$(dirname "$0")"
echo "正在启动… 浏览器会自动打开 http://127.0.0.1:8777"
echo "（关掉这个窗口就停止服务）"
echo
exec python3 site/serve.py --open --port 8777
