#!/usr/bin/env bash
# 依次运行全部 demo。用法: bash run_all.sh  或  bash run_all.sh 03
#
# 想调参、想边跑边逐条验证结论, 用网站更方便:
#     python3 site/serve.py --open
set -u
cd "$(dirname "$0")"

filter="${1:-}"
fail=0

for f in $(ls 0*/demo*.py | sort); do
  case "$f" in
    ${filter}*) ;;
    *) [ -n "$filter" ] && continue ;;
  esac
  echo
  echo "################################################################################"
  echo "### $f"
  echo "################################################################################"
  if ! python3 -u "$f"; then
    echo ">>> 运行失败: $f"
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "全部 demo 运行完毕。"
else
  echo "有 demo 运行失败，见上面的输出。"
  exit 1
fi
