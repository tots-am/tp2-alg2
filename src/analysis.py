"""Análise dos resultados: complexidade empírica, warm-start e sensibilidade.

  python -m src.analysis            # gera results/analysis/{figs,tables}

Funciona com dados parciais (o B&B pode ainda estar rodando): cada seção usa o
que existir em results/ e ignora o resto.
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402

from .tsp import instances  # noqa: E402

RES = os.path.join(instances.ROOT, "results")
OUT = os.path.join(RES, "analysis")
FIG = os.path.join(OUT, "figs")
TAB = os.path.join(OUT, "tables")

# Paleta categórica (ordem fixa, validada) + marcador por estratégia (codificação
# secundária: identidade nunca depende só da cor; funciona impresso em P&B).
STRATS = ["trivial", "nn", "hull", "tat", "christofides"]
LABEL = {"trivial": "Trivial", "nn": "Viz. mais próximo", "hull": "Fecho convexo",
         "tat": "Twice-around-tree", "christofides": "Christofides"}
COLOR = dict(zip(STRATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARK = dict(zip(STRATS, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "lines.markersize": 5, "legend.frameon": False,
})


# ============================================================================
# utilidades
# ============================================================================
def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def _table(df: pd.DataFrame, name: str, floatfmt="{:.3g}", index=True):
    """Salva CSV e um tabular LaTeX simples (sem depender de jinja2)."""
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "--" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("_", r"\_").replace("%", r"\%")

    lines = [r"\begin{tabular}{" + "l" * 1 + "r" * (d.shape[1] - 1) + "}", r"\hline",
             " & ".join(fmt(c) for c in d.columns) + r" \\", r"\hline"]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in d.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _metrics(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, float(np.sqrt(ss_res / len(y)))


# ============================================================================
# 1. Complexidade empírica das heurísticas / sub-rotinas
# ============================================================================
CANDIDATES = {
    "n": lambda n: n,
    "n log n": lambda n: n * np.log(n),
    "n^2": lambda n: n ** 2,
    "n^2 log n": lambda n: n ** 2 * np.log(n),
    "n^3": lambda n: n ** 3,
}
# (estratégia, coluna, rótulo, complexidade teórica esperada)
SERIES = [
    ("nn", "t_nn_walk", "Viz. mais próximo", "n^2"),
    ("hull", "t_insertion", "Fecho: inserção", "n^3"),
    ("tat", "t_mst", "MST (Prim denso)", "n^2"),
    ("tat", "time_median", "Twice-around-tree (total)", "n^2"),
    ("christofides", "t_matching", "Christofides: matching", "n^3"),
    ("christofides", "t_matching_build_graph", "Christofides: grafo do matching", "n^2"),
    ("christofides", "time_median", "Christofides (total)", "n^3"),
]


def fit_curves(n, y):
    """Ajusta y = a*f(n) + c (a, c >= 0, NNLS) para cada candidato e a lei de
    potência log y = log a + b log n. Retorna lista de dicts com R² e RMSE."""
    out = []
    for name, f in CANDIDATES.items():
        X = np.column_stack([f(n), np.ones_like(n)])
        scale = X.max(axis=0)
        coef, _ = nnls(X / scale, y)
        yhat = (X / scale) @ coef
        r2, rmse = _metrics(y, yhat)
        out.append({"model": name, "a": coef[0] / scale[0], "c": coef[1] / scale[1],
                    "R2": r2, "RMSE": rmse})
    m = y > 0
    b, loga, r, _, se = stats.linregress(np.log(n[m]), np.log(y[m]))
    yhat = np.exp(loga) * n ** b
    r2, rmse = _metrics(y, yhat)
    out.append({"model": f"potência (b={b:.2f}±{1.96 * se:.2f})", "a": np.exp(loga), "c": 0.0,
                "R2": r2, "RMSE": rmse, "b": b, "b_ci": 1.96 * se, "R2_log": r ** 2})
    return out


def complexity():
    path = os.path.join(RES, "scaling.csv")
    if not os.path.exists(path):
        print("[complexidade] scaling.csv ausente — rode `python -m src.run_experiments scaling`")
        return
    sc = pd.read_csv(path)
    rows, best = [], []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    marks = ["o", "s", "^", "D", "v", "P", "X"]
    for (strat, col, label, theo), c, mk in zip(SERIES, colors, marks):
        d = sc[sc.strategy == strat].sort_values("n")
        if col not in d or d[col].isna().all():
            continue
        n, y = d.n.to_numpy(float), d[col].to_numpy(float)
        fits = fit_curves(n, y)
        for f in fits:
            rows.append({"série": label, "teórica": theo, **{k: f[k] for k in ("model", "R2", "RMSE")}})
        poly = [f for f in fits if not f["model"].startswith("pot")]
        top = max(poly, key=lambda f: f["R2"])
        pw = fits[-1]
        best.append({"série": label, "teórica": theo, "melhor ajuste": top["model"],
                     "R2": top["R2"], "RMSE (s)": top["RMSE"],
                     "expoente b": pw["b"], "IC95% b": pw["b_ci"], "R2 (log)": pw["R2_log"]})
        ax = axes[0] if "total" not in label else axes[1]
        ax.loglog(n, y, mk, color=c, label=label, markersize=4)
        nn_ = np.linspace(n.min(), n.max(), 200)
        ax.loglog(nn_, top["a"] * CANDIDATES[top["model"]](nn_) + top["c"], "-", color=c, lw=1)
    for ax, t in zip(axes, ["Sub-rotinas", "Algoritmo completo"]):
        ax.set_xlabel("N (vértices)")
        ax.set_ylabel("tempo (s)")
        ax.set_title(t, color=INK, fontsize=9, loc="left")
        ax.legend(fontsize=7)
    _save(fig, "complexidade_tempo")
    _table(pd.DataFrame(rows).set_index("série"), "complexidade_todos_ajustes")
    _table(pd.DataFrame(best).set_index("série"), "complexidade_melhor_ajuste")

    # memória (pico tracemalloc) — o grafo NetworkX do matching é O(k^2) objetos Python
    mem_rows = []
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for (strat, col, label), c, mk in zip(
            [("tat", "mem_mst", "MST (NumPy)"), ("hull", "mem_insertion", "Fecho: inserção"),
             ("christofides", "mem_matching_build_graph", "Grafo do matching (NetworkX)"),
             ("christofides", "mem_matching", "Matching (NetworkX)")],
            ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4"], ["o", "^", "s", "v"]):
        d = sc[sc.strategy == strat].sort_values("n")
        n, y = d.n.to_numpy(float), d[col].to_numpy(float) / 1024
        pw = fit_curves(n, y)[-1]
        mem_rows.append({"sub-rotina": label, "expoente b": pw["b"], "IC95% b": pw["b_ci"],
                         "R2 (log)": pw["R2_log"], "pico em N máx (KB)": y[-1]})
        ax.loglog(n, y, mk + "-", color=c, label=label, markersize=4, lw=1)
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("pico de memória (KB)")
    ax.legend(fontsize=7)
    _save(fig, "complexidade_memoria")
    _table(pd.DataFrame(mem_rows).set_index("sub-rotina"), "complexidade_memoria")
    print(pd.DataFrame(best).set_index("série").round(4).to_string())
    print(pd.DataFrame(mem_rows).set_index("sub-rotina").round(3).to_string())


# ============================================================================
# 2. Branch-and-Bound: crescimento e impacto do warm-start
# ============================================================================
def _load_bnb():
    path = os.path.join(RES, "bnb.csv")
    if not os.path.exists(path):
        return None
    b = pd.read_csv(path)
    b["solved"] = b.status == "optimal"
    b["prune_rate"] = b.nodes_pruned / (b.nodes_expanded + b.nodes_pruned)
    return b


def reference_optimum(b: pd.DataFrame | None) -> pd.DataFrame:
    """Ótimo de referência por instância: custo de qualquer execução que provou
    otimalidade; senão ótimo publicado; senão melhor custo achado (marcado)."""
    rows = []
    for name, k in instances.TEST_SET:
        inst = instances.load(name, k)
        opt, src = None, None
        if b is not None:
            s = b[(b.instance == inst.name) & b.solved]
            if len(s):
                opt, src = int(s.cost.min()), "B&B"
        if opt is None and inst.known_opt:
            opt, src = inst.known_opt, "TSPLIB"
        if opt is None and b is not None:
            s = b[b.instance == inst.name]
            if len(s):
                opt, src = int(s.cost.min()), "melhor achado"
        rows.append({"instance": inst.name, "n": inst.n, "opt": opt, "opt_source": src})
    return pd.DataFrame(rows)


def bnb_growth(b):
    s = b[b.solved & (b.strategy == "christofides")]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for bound, c, mk in [("hk", "#2a78d6", "o"), ("mst", "#eb6834", "s")]:
        d = s[s.bound == bound]
        na = b[(b.bound == bound) & (b.strategy == "christofides") & ~b.solved]
        for ax, col, lab in [(axes[0], "nodes_expanded", "nós expandidos"), (axes[1], "bnb_time", "tempo (s)")]:
            ax.semilogy(d.n, d[col], mk, color=c, label=f"cota {bound}")
            if len(na):
                ax.semilogy(na.n, na[col], mk, mfc="none", color=c, label=f"cota {bound}: NA (censurado)")
            if len(d) >= 3:
                y = np.log10(d[col].clip(lower=1e-6))
                sl, ic, r, _, se = stats.linregress(d.n, y)
                xx = np.linspace(d.n.min(), d.n.max(), 50)
                ax.semilogy(xx, 10 ** (ic + sl * xx), "-", color=c, lw=1)
                rows.append({"cota": bound, "medida": lab, "pontos": len(d),
                             "fator por vértice": 10 ** sl, "R2 (log)": r ** 2,
                             "RMSE (log10)": float(np.sqrt(((y - (ic + sl * d.n)) ** 2).mean()))})
            ax.set_xlabel("N (vértices)")
            ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Warm-start: Christofides", color=INK, fontsize=9, loc="left")
    _save(fig, "bnb_crescimento")
    if rows:
        _table(pd.DataFrame(rows).set_index("cota"), "bnb_crescimento")
        print(pd.DataFrame(rows).round(4).to_string(index=False))


def warmstart(b, ref):
    b = b.merge(ref[["instance", "opt"]], on="instance", how="left")
    b["ub0_gap"] = b.initial_ub / b.opt - 1
    b["final_gap"] = b.cost / b.opt - 1
    triv = b[b.strategy == "trivial"].set_index(["instance", "bound"])
    key = list(zip(b.instance, b.bound))
    b["nodes_vs_trivial"] = b.nodes_expanded.to_numpy() / triv.nodes_expanded.reindex(key).to_numpy()
    b["speedup_total"] = triv.total_time.reindex(key).to_numpy() / b.total_time.to_numpy()

    cols = ["instance", "n", "bound", "strategy", "status", "ub0_gap", "nodes_expanded", "nodes_pruned",
            "prune_rate", "heur_time", "bnb_time", "total_time", "nodes_vs_trivial", "speedup_total",
            "final_gap"]
    _table(b[cols].sort_values(["bound", "n", "strategy"]), "warmstart_detalhado", index=False)

    # resumo por cota x estratégia (médias geométricas sobre instâncias resolvidas por todas)
    summ = []
    for bound, g in b.groupby("bound"):
        full = g.groupby("instance").solved.all()
        ok = full[full].index
        for s in STRATS:
            d = g[(g.strategy == s) & g.instance.isin(ok)]
            dd = g[g.strategy == s]
            if not len(dd):
                continue
            summ.append({"cota": bound, "estratégia": LABEL[s],
                         "resolvidas": f"{int(dd.solved.sum())}/{len(dd)}",
                         "gap inicial médio": dd.ub0_gap.mean(),
                         "nós vs trivial (geo)": stats.gmean(d.nodes_vs_trivial) if len(d) else np.nan,
                         "speedup total (geo)": stats.gmean(d.speedup_total) if len(d) else np.nan,
                         "taxa de poda": dd.prune_rate.mean(),
                         "tempo heurística (ms)": 1e3 * dd.heur_time.mean()})
    sm = pd.DataFrame(summ).set_index("cota")
    _table(sm, "warmstart_resumo")
    print(sm.round(4).to_string())

    # figura: nós expandidos por instância, uma coluna por estratégia, painel por cota
    bounds = [x for x in ("hk", "mst") if x in set(b.bound)]
    fig, axes = plt.subplots(len(bounds), 1, figsize=(9, 3.0 * len(bounds)), squeeze=False)
    order = ref.sort_values("n").instance.tolist()
    for ax, bound in zip(axes[:, 0], bounds):
        g = b[b.bound == bound]
        x = np.arange(len(order))
        w = 0.16
        for i, s in enumerate(STRATS):
            d = g[g.strategy == s].set_index("instance").reindex(order)
            xs = x + (i - 2) * w
            ok = d.solved.fillna(False).astype(bool).to_numpy()
            ax.bar(xs[ok], d.nodes_expanded.to_numpy()[ok], w * 0.88, color=COLOR[s], label=LABEL[s],
                   edgecolor="white", linewidth=0.5)
            ax.bar(xs[~ok], d.nodes_expanded.to_numpy()[~ok], w * 0.88, color="white",
                   edgecolor=COLOR[s], hatch="////", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("nós expandidos")
        ax.set_title(f"Cota {bound} (hachurado = NA, contagem até o corte)", color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    # legenda com handles explícitos: a cor segue a estratégia mesmo sem barras no painel
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[s], label=LABEL[s]) for s in STRATS]
    handles.append(Patch(facecolor="white", edgecolor=INK2, hatch="////", label="NA (tempo esgotado)"))
    axes[0, 0].legend(handles=handles, ncol=6, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.25))
    _save(fig, "warmstart_nos")

    # figura: trade-off — gap inicial vs redução de nós
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for s in STRATS:
        d = b[(b.strategy == s) & b.solved & (b.bound == bounds[0])]
        ax.semilogy(d.ub0_gap * 100, d.nodes_vs_trivial, MARK[s], color=COLOR[s], label=LABEL[s])
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("gap do limite superior inicial (%)")
    ax.set_ylabel("nós expandidos / trivial")
    ax.set_title(f"Cota {bounds[0]}", color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7)
    _save(fig, "warmstart_tradeoff")


# ============================================================================
# 3. Sensibilidade: features x gap de aproximação
# ============================================================================
FEATS = ["n", "aspect_ratio", "hull_ratio", "spatial_entropy", "clark_evans_R", "nn_cv",
         "mst_max_degree", "mst_odd_frac", "mst_edge_cv", "dist_cv"]


def sensitivity(ref):
    h = pd.read_csv(os.path.join(RES, "heuristics.csv"))
    h = h[h.set == "test"].merge(ref, on=["instance", "n"])
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    h["gap"] = h.cost / h.opt - 1
    d = h[h.strategy != "trivial"].merge(f.drop(columns=["n"]), on="instance")
    d = d.dropna(subset=["gap"])
    nref = ref.opt_source.value_counts().to_dict()
    print(f"[sensibilidade] fonte do ótimo de referência: {nref}")
    if d.instance.nunique() < 6:
        print(f"[sensibilidade] só {d.instance.nunique()} instâncias com ótimo — aguardando o B&B")
        return

    gaps = d.pivot(index="instance", columns="strategy", values="gap")[["nn", "hull", "tat", "christofides"]]
    _table(gaps.join(ref.set_index("instance")[["n", "opt", "opt_source"]]).sort_values("n"), "gaps")

    # correlação de Spearman (não-paramétrica, robusta a 15 pontos e outliers)
    rho = pd.DataFrame(index=FEATS, columns=gaps.columns, dtype=float)
    pv = rho.copy()
    fi = f.set_index("instance").loc[gaps.index]
    for s in gaps.columns:
        for ft in FEATS:
            r, p = stats.spearmanr(fi[ft], gaps[s])
            rho.loc[ft, s], pv.loc[ft, s] = r, p
    _table(rho.rename(columns=LABEL), "spearman_rho")
    _table(pv.rename(columns=LABEL), "spearman_p")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(rho.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rho.columns)))
    ax.set_xticklabels([LABEL[c] for c in rho.columns], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels(FEATS, fontsize=7)
    for i in range(len(FEATS)):
        for j in range(len(rho.columns)):
            r, p = rho.iloc[i, j], pv.iloc[i, j]
            ax.text(j, i, f"{r:.2f}" + ("*" if p < 0.05 else ""), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(r) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="ρ de Spearman (* p<0,05)")
    _save(fig, "sensibilidade_spearman")

    # Modelos agrupados (4 algoritmos x instâncias), validados por leave-one-out.
    #  "bruto":    alvo = gap, com o algoritmo como variável dummy -> o modelo
    #              aprende sobretudo "qual algoritmo" (resposta trivial);
    #  "centrado": alvo = gap - gap médio do algoritmo, só features -> responde
    #              "que instância é mais difícil que o normal para qualquer algoritmo".
    # Obs.: CV por instância (LeaveOneGroupOut) evita que as 4 linhas da mesma
    # instância vazem entre treino e teste.
    groups = d.instance.to_numpy()
    d = d.assign(gap_c=d.gap - d.groupby("strategy").gap.transform("mean"))
    variants = {
        "bruto": (pd.get_dummies(d[FEATS + ["strategy"]], columns=["strategy"], dtype=float), d.gap.to_numpy()),
        "centrado": (d[FEATS].astype(float), d.gap_c.to_numpy()),
    }
    res = []
    for vname, (X, y) in variants.items():
        cv = LeaveOneGroupOut()
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, random_state=0)
        yt = cross_val_predict(tree, X, y, cv=cv, groups=groups)
        tree.fit(X, y)
        res.append({"modelo": f"Árvore prof. 3 ({vname})", "R2 treino": tree.score(X, y),
                    "R2 CV": _metrics(y, yt)[0], "RMSE CV": _metrics(y, yt)[1]})
        imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
        _table(imp[imp > 0].to_frame("importância"), f"arvore_importancia_{vname}")
        with open(os.path.join(TAB, f"arvore_regras_{vname}.txt"), "w") as fh:
            fh.write(export_text(tree, feature_names=list(X.columns)))

        Xs = StandardScaler().fit_transform(X)
        lr = RidgeCV(alphas=np.logspace(-2, 3, 30))   # penalização L2: 10 features, ~15 instâncias
        yl = cross_val_predict(lr, Xs, y, cv=cv, groups=groups)
        lr.fit(Xs, y)
        res.append({"modelo": f"Regressão Ridge ({vname})", "R2 treino": lr.score(Xs, y),
                    "R2 CV": _metrics(y, yl)[0], "RMSE CV": _metrics(y, yl)[1]})
        coef = pd.Series(lr.coef_, index=X.columns)
        _table(coef.reindex(coef.abs().sort_values(ascending=False).index).to_frame("coef. padronizado"),
               f"regressao_coeficientes_{vname}")
        print(f"importâncias árvore ({vname}):  "
              + ", ".join(f"{k}={v:.2f}" for k, v in imp[imp > 0].items()))
    res = pd.DataFrame(res).set_index("modelo")
    _table(res, "sensibilidade_modelos")
    print(res.round(3).to_string())


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print("== 1. Complexidade empírica")
        complexity()
        b = _load_bnb()
        ref = reference_optimum(b)
        if b is not None and len(b):
            print(f"== 2. Branch-and-Bound ({len(b)} execuções)")
            bnb_growth(b)
            warmstart(b, ref)
        print("== 3. Sensibilidade")
        sensitivity(ref)
    print(f"saída em {OUT}")


if __name__ == "__main__":
    main()
