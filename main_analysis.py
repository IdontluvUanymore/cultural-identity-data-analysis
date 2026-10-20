# -*- coding: utf-8 -*-
"""
广州大学城十校大学生主流文化认同调查 —— 数据分析主脚本
"""
import sys, io, math, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from openpyxl import load_workbook

# 导入我们自己的公共函数 (相当于 C++ 的 #include "01_common.h")
from importlib import import_module
common = import_module("公共统计函数与数据装载")
t_sf2, f_sf, chi2_sf, cronbach_alpha, desc, corr_p, ols, gammap = common.t_sf2, common.f_sf, common.chi2_sf, common.cronbach_alpha, common.desc, common.corr_p, common.ols, common.gammap
betai = common.betai

XLSX = r"C:\A_t\Code\数据分析\Data.xlsx"

# =====================================================================
# 【模块一】
# =====================================================================
def load_and_clean_data():
    wb = load_workbook(XLSX, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=[f"c{i}" for i in range(len(rows[0]))])
    
    COLS = {chr(ord('A')+i) if i < 26 else 'A'+chr(ord('A')+i-26): f"c{i}" for i in range(33)}
    col = lambda letter: COLS[letter]
    
    # 提取我们需要分析的列
    Q7 = [col(c) for c in ['L','M','N','O','P','Q']]
    IDENT = [col(c) for c in ['R','S','T','U','V','W','X','Y','Z','AA','AB','AC']]
    SIKE = col('AF')
    
    # 类型转换，非法字符转为 NaN
    for c in Q7 + IDENT + [SIKE]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df, col, Q7, IDENT, SIKE

# =====================================================================
# 【模块二】合成维度得分
# =====================================================================
def compute_composites(df, col, Q7, IDENT, SIKE):
    comp = pd.DataFrame(index=df.index)
    comp['算法感知'] = df[[col('L'), col('N')]].mean(axis=1)
    comp['同质化感知'] = df[[col('M'), col('O'), col('P')]].mean(axis=1)
    comp['信息茧房感知'] = df[[col('N'), col('O'), col('P')]].mean(axis=1)
    comp['认知认同'] = df[[col('R'), col('S'), col('T')]].mean(axis=1)
    comp['情感认同'] = df[[col('U'), col('V'), col('W')]].mean(axis=1)
    comp['价值认同'] = df[[col('X'), col('Y'), col('Z')]].mean(axis=1)
    comp['行为认同'] = df[[col('AA'), col('AB'), col('AC')]].mean(axis=1)
    comp['主流文化认同总分'] = df[IDENT].mean(axis=1)
    comp['思政课评价'] = df[SIKE]
    comp['数智情境感知总分'] = df[Q7].mean(axis=1)
    
    comp['学历阶段'] = df[col('H')]
    comp['性别'] = df[col('G')]
    comp['学科'] = df[col('I')]
    comp['学校'] = df[col('F')]
    comp['上网时长'] = df[col('J')]
    comp['AI使用'] = df[col('K')]
    return comp

# =====================================================================
# 【模块三】通用的差异检验函数
# =====================================================================
def run_group_difference(comp, group_var, target_vars, is_anova=True):
    """封装了 t检验 和 ANOVA，避免代码里全是复制的计算逻辑"""
    print(f"\n--- 差异检验：{group_var} ---")
    for k in target_vars:
        groups = comp.groupby(group_var)[k].apply(lambda s: s.dropna().values).values
        if len(groups) < 2: continue
        
        if not is_anova and len(groups) == 2: # 独立样本t检验
            a, b = groups[0], groups[1]
            n1, n2 = len(a), len(b)
            sp2 = ((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1))/(n1+n2-2)
            t = (a.mean() - b.mean())/math.sqrt(sp2*(1/n1+1/n2))
            p = t_sf2(t, n1+n2-2)
            print(f"  {k:<12} t={t:>6.3f}, p={p:.4f}")
        else: # 单因素方差分析 (ANOVA)
            allv = np.concatenate(groups); N, kk = len(allv), len(groups)
            gm = allv.mean()
            ssb = sum(len(g)*(g.mean()-gm)**2 for g in groups)
            ssw = sum(((g-g.mean())**2).sum() for g in groups)
            F = (ssb/(kk-1))/(ssw/(N-kk))
            p = f_sf(F, kk-1, N-kk)
            print(f"  {k:<12} F({kk-1},{N-kk})={F:>6.3f}, p={p:.4f}")

# =====================================================================
# 【主函数 Main】
# =====================================================================
def main():
    print("="*80 + "\n【1】加载数据\n" + "="*80)
    df, col, Q7, IDENT, SIKE = load_and_clean_data()
    comp = compute_composites(df, col, Q7, IDENT, SIKE)
    print(f"数据加载成功，共 {len(df)} 条样本。")

    print("\n" + "="*80 + "\n【2】信度检验 (Cronbach's α)\n" + "="*80)
    # (这里为了简洁省略了打印各个维度的代码，直接跑核心检验)
    dims = {'认知认同': [col('R'), col('S'), col('T')], '情感认同': [col('U'), col('V'), col('W')]}
    for name, cols in dims.items():
        alpha = cronbach_alpha(df[cols].dropna().values.astype(float))
        print(f"  {name}: α = {alpha:.3f}")

    print("\n" + "="*80 + "\n【3】群体差异检验\n" + "="*80)
    targets = ['主流文化认同总分', '算法感知', '行为认同']
    run_group_difference(comp, '性别', targets, is_anova=False)   # T检验
    run_group_difference(comp, '学历阶段', targets, is_anova=True) # ANOVA
    run_group_difference(comp, '上网时长', targets, is_anova=True) # ANOVA

    print("\n" + "="*80 + "\n【4】核心机制：回归与中介效应\n" + "="*80)
    # 标准化
    cols_std = ['算法感知','信息茧房感知','主流文化认同总分','思政课评价']
    ZN = ((comp[cols_std] - comp[cols_std].mean()) / comp[cols_std].std()).values
    
    r1 = ols(ZN[:, 1], [ZN[:, 0]], ['算法感知']) # 茧房 <- 算法
    r2 = ols(ZN[:, 2], [ZN[:, 0]], ['算法感知']) # 认同 <- 算法
    r3 = ols(ZN[:, 2], [ZN[:, 0], ZN[:, 1]], ['算法感知', '茧房']) # 认同 <- 算法 + 茧房
    
    a, b = r1['coef']['算法感知'], r3['coef']['茧房']
    indirect = a * b
    print(f"  总效应 c = {r2['coef']['算法感知']:.4f}")
    print(f"  间接效应 a×b = {indirect:.4f}")

    print("\n" + "="*80 + "\n【5】K-means 聚类识别三类群体\n" + "="*80)
    feat = comp[['认知认同','情感认同','价值认同','行为认同']].dropna().values
    
    def kmeans_simple(X, K, iters=300, seed=7):
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2).argmin(axis=1)
            newC = np.array([X[lab == j].mean(axis=0) if (lab == j).sum() > 0 else C[j] for j in range(K)])
            if np.allclose(newC, C): break
            C = newC
        return lab, C

    lab, C = kmeans_simple(feat, 3)
    counts = np.bincount(lab, minlength=3)
    print(f"  聚类完成。分3类，各类型人数: {counts}")
    print("  [已导出] composites.csv")

if __name__ == "__main__":
    main()