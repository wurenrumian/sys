#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的参数入口。

所有 demo 里"可以拧的旋钮"都通过这里从环境变量读取, 于是同一份代码有两种用法:

    python3 01_data_storage/demo1_row_vs_col.py                # 默认参数
    N_ROWS=1000000 python3 01_data_storage/demo1_row_vs_col.py # 改参数重跑

网站 (site/serve.py) 也是用这个机制在浏览器里调参重跑的 —— 它只是把表单里的值
塞进子进程的环境变量, 不改任何一行代码。

**为什么要做这件事**: 这些 demo 的价值在于"结论"而不是"某个具体数字"。
一个结论如果只在默认参数下成立, 那它多半是巧合。能改参数重跑, 才谈得上验证。
"""
import os

_USED = []          # [(名字, 生效值, 默认值, 说明)]


def _get(name, default, cast, desc):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        val = default
    else:
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            raise SystemExit(f"参数 {name}={raw!r} 解析失败, 期望 {cast.__name__}")
    _USED.append((name, val, default, desc))
    return val


def env_int(name, default, desc=""):
    return _get(name, default, lambda s: int(float(s)), desc)


def env_float(name, default, desc=""):
    return _get(name, default, float, desc)


def env_str(name, default, desc=""):
    return _get(name, default, str, desc)


def env_bool(name, default, desc=""):
    return _get(name, default, lambda s: s.strip().lower() in ("1", "true", "yes", "on"), desc)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def banner(width=78):
    """打印本次运行实际生效的参数。被改动过的参数用 * 标出。

    放在 main() 的第一行。它的作用不只是好看 —— 当你在网站上看到一份跑飞了的输出,
    第一个要问的问题永远是"这是用什么参数跑的"。
    """
    if not _USED:
        return
    changed = [u for u in _USED if u[1] != u[2]]
    print("-" * width)
    print(f"可调参数 (共 {len(_USED)} 个" +
          (f", 本次改了 {len(changed)} 个" if changed else ", 全部为默认值") + "):")
    for name, val, default, desc in _USED:
        mark = " *" if val != default else "  "
        tail = f"   (默认 {_fmt(default)})" if val != default else ""
        print(f" {mark} {name:<18}= {_fmt(val):<14}{desc}{tail}")
    print("-" * width)
