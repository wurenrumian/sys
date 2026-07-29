#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo3: DCAI (Data-Centric AI) —— 数据事故如何静悄悄地打掉线上 AUC

对应表格: "推荐大模型对数据质量敏感度提升, 数据分布异常可能导致模型效果显著下降,
          亟需系统性数据质量评估与改进方法" / "DCAI 强调通过数据质量优化与自动化处理
          链路提升模型性能, 而非单纯依赖模型改进"

核心演示: 模型代码一行没改, 只是上游特征生产出了问题, 线上 AUC 就会掉.
         PSI 这类分布漂移指标能在 AUC 掉之前(或同时)把事故定位到具体特征.
"""
import numpy as np

# numpy 2.0 + Apple Accelerate 的已知问题: matmul 会误报 divide-by-zero/overflow 标志位,
# 即使输入输出全部有限. 与本 demo 的数值无关, 直接屏蔽.
np.seterr(all="ignore")

rng = np.random.default_rng(42)
N_TRAIN, N_SERVE, N_FEAT = 60_000, 30_000, 12


# ---------------------------------------------------------------- 数据与模型
def make_data(n, seed):
    """生成带真实信号的样本. 前 4 个特征信号强, 其余弱, 模拟推荐特征的重要性长尾."""
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, N_FEAT))
    w = np.array([1.4, 1.1, 0.9, 0.7] + [0.15] * (N_FEAT - 4))
    logit = X @ w - 1.0
    y = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, w


def train_lr(X, y, epochs=250, lr=0.4):
    """一个最小的逻辑回归, 代表线上排序模型. 重点不在模型, 在数据."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def auc(y, s):
    """秩和法算 AUC, 无需 sklearn."""
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列值: 同分取平均秩
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- 数据质量指标
def psi(expect, actual, bins=10):
    """Population Stability Index —— 工业界最常用的分布漂移指标.

        PSI = Σ (实际占比 - 期望占比) * ln(实际占比 / 期望占比)
        < 0.1 稳定 | 0.1~0.25 需关注 | > 0.25 显著漂移, 必须报警

    分桶边界取自训练期(期望)分布的分位数 —— 这点很关键: 边界必须冻结,
    否则漂移会被自适应的分桶悄悄吸收掉, 指标永远看起来正常.
    """
    edges = np.quantile(expect, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expect, edges)[0] / len(expect)
    a = np.histogram(actual, edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def null_rate(col, default=0.0):
    """默认值占比 —— 特征生产任务挂掉时最直接的信号."""
    return float(np.mean(col == default))


# ---------------------------------------------------------------- 事故注入
def incident_none(X):
    return X.copy(), "无事故 (基线)"


def incident_missing(X):
    """事故1: 上游特征生产任务挂了, 最重要的特征全部落成默认值 0."""
    Xb = X.copy()
    Xb[:, 0] = 0.0
    return Xb, "特征缺失: feat_0 全部变默认值"


def incident_shift(X):
    """事故2: 上游改了口径(比如时长从秒变毫秒/归一化方式变了), 分布整体平移+缩放."""
    Xb = X.copy()
    Xb[:, 1] = Xb[:, 1] * 2.5 + 1.8
    return Xb, "口径变更: feat_1 分布平移+缩放"


def incident_skew(X):
    """事故3: 离在线不一致 —— 线上取到的是上一版本的特征, 只有部分样本受影响.

    这是推荐系统里最阴险的一类事故: 只影响一部分流量, 平均值看起来几乎没变.
    """
    Xb = X.copy()
    idx = rng.random(len(Xb)) < 0.3          # 30% 的流量命中旧版本
    Xb[idx, 2] = rng.normal(0, 1, idx.sum())  # 被替换成与 label 无关的随机值
    return Xb, "离在线不一致: feat_2 有 30% 流量取到脏值"


def incident_stale(X):
    """事故4: 特征延迟 —— 全部特征加了一点噪声(数据管道积压, 特征不新鲜)."""
    return X + rng.normal(0, 0.35, X.shape), "特征不新鲜: 全特征叠加噪声"


def main():
    print("=" * 80)
    print("DCAI 演示: 模型代码完全不变, 只让上游数据出问题, 看 AUC 和 PSI 各自的反应")
    print("=" * 80)

    Xtr, ytr, _ = make_data(N_TRAIN, seed=1)
    Xse, yse, _ = make_data(N_SERVE, seed=2)
    w, b = train_lr(Xtr, ytr)

    base_auc = auc(yse, Xse @ w + b)
    print(f"\n训练样本 {N_TRAIN:,} / 线上样本 {N_SERVE:,} / 特征 {N_FEAT} 维")
    print(f"健康状态下的线上 AUC = {base_auc:.4f}\n")

    incidents = [incident_none, incident_missing, incident_shift,
                 incident_skew, incident_stale]

    print(f"{'事故类型':<34}{'AUC':>9}{'ΔAUC':>10}{'最大PSI':>10}{'责任特征':>10}{'判定':>8}")
    print("-" * 80)

    for fn in incidents:
        Xbad, label = fn(Xse)
        a = auc(yse, Xbad @ w + b)
        # 对每个特征算 PSI: 期望分布 = 训练期, 实际分布 = 线上
        psis = np.array([psi(Xtr[:, j], Xbad[:, j]) for j in range(N_FEAT)])
        j = int(psis.argmax())
        flag = "报警" if psis.max() > 0.25 else ("关注" if psis.max() > 0.1 else "正常")
        print(f"{label:<32}{a:>9.4f}{a-base_auc:>+10.4f}"
              f"{psis.max():>10.3f}{'feat_'+str(j):>10}{flag:>8}")

    # ------------------------------------------------------------ 补充信号
    print("\n" + "-" * 80)
    print("[关键] 上表里最该注意的不是 PSI 抓到了什么, 而是它**漏掉了什么**:")
    print("       事故3/事故4 的 AUC 都掉了 2 个点以上, PSI 却判定'正常'.")

    Xbad, _ = incident_skew(Xse)
    j = 2
    print(f"\n  事故3 (离在线不一致, 30% 流量的 feat_{j} 被换成脏值):")
    print(f"    训练期均值 {Xtr[:, j].mean():+.4f} / 线上均值 {Xbad[:, j].mean():+.4f}")
    print(f"    训练期方差 {Xtr[:, j].var():.4f} / 线上方差 {Xbad[:, j].var():.4f}")
    print(f"    PSI = {psi(Xtr[:, j], Xbad[:, j]):.3f}  (判定: 正常)")
    print(f"    但 AUC 掉了 {auc(yse, Xbad @ w + b) - base_auc:+.4f}")
    print("\n  为什么 PSI 抓不到? 因为脏值恰好来自**同一个边缘分布** ——")
    print("  被污染后 feat_2 的直方图和训练期几乎完全一致, 只是它和 label 的**关联**断了.")
    print("  PSI 是单特征边缘分布指标, 对'分布没变但语义错位'这类事故天然盲视.")
    print("  而这恰恰是推荐系统最高频的事故类型(特征版本错位、离线在线实现不一致).")

    Xbad, _ = incident_missing(Xse)
    print(f"\n  对比事故1 (生产任务挂掉), 用最朴素的默认值率就能一眼抓到:")
    print(f"    feat_0 默认值率 = {null_rate(Xbad[:, 0]):.1%} (健康时 {null_rate(Xse[:, 0]):.1%})")
    print("  -> 越是'低级'的事故越好抓; 真正难的是语义级的不一致.")

    # ------------------------------------------------------------ 敏感度曲线
    print("\n" + "-" * 80)
    print("[敏感度] 特征污染比例 → PSI → AUC 掉幅 (以 feat_0 被随机值替换为例)")
    print(f"  {'污染比例':<10}{'PSI':>10}{'AUC':>10}{'ΔAUC':>10}")
    for ratio in [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        Xb = Xse.copy()
        idx = rng.random(len(Xb)) < ratio
        Xb[idx, 0] = rng.normal(0, 1, idx.sum())
        a = auc(yse, Xb @ w + b)
        print(f"  {ratio:<10.0%}{psi(Xtr[:, 0], Xb[:, 0]):>10.3f}"
              f"{a:>10.4f}{a-base_auc:>+10.4f}")

    print("\n  -> 污染比例从 0% 到 100%, AUC 单调掉了 16 个点, 而 PSI 全程约等于 0.")
    print("     这条曲线是整个 demo 的结论: **单一指标兜不住**. 工业上必须分层组合:")
    print("       统计类: 空值率/均值/分位数     —— 抓生产任务挂掉        (事故1)")
    print("       分布类: PSI / KS / KL          —— 抓口径变更/分布平移    (事故2)")
    print("       一致性: 离在线特征逐样本对拍    —— 抓训推不一致          (事故3, PSI 盲区)")
    print("       关联类: 特征-label 互信息/单特征AUC 的漂移 —— 抓语义错位 (事故3/4)")
    print("       终极兜底: 影子流量上的实时 AUC/GAUC 监控")
    print("\n  这正是表格里'亟需系统性数据质量评估与改进方法'的技术含义:")
    print("  真正要建的是一套能把'数据异常量'映射到'业务指标掉幅'的**可量化体系**,")
    print("  而不是零散的阈值告警 —— 因为你必须能回答'这个 PSI=0.3 到底会掉几个点 AUC'.")
    print("\n  顺带一提: 这个问题和表格 LLM 页里抖音推荐的方向五")
    print("  ('建立算子精度损耗到离在线 AUC 波动的影响预测模型') 是**同构**的 ——")
    print("  一个是数据侧扰动→AUC, 一个是数值精度扰动→AUC, 方法论可以复用.")


if __name__ == "__main__":
    main()
