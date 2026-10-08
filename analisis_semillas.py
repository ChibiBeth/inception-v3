#!/usr/bin/env python3
"""Resumen de los experimentos con varias semillas y análisis de las señas parecidas.

Lee lo que generan experimentos_semillas.sh y evaluate_metrics.py:
  metricas/semillas/<experimento>/s<semilla>/      (modelos evaluados sobre test)
  metricas/semillas_cv/<experimento>/s<semilla>/   (validación cruzada por persona)

y escribe en --out (por defecto metricas/analisis_semillas/):
  1. Media ± desvío estándar entre semillas de exactitud, F1 macro y kappa
     (tabla_semillas_test.tex, tabla_semillas_cv.tex, semillas_*.csv).
  2. Análisis de las señas parecidas (pares definidos en PARES):
       - exactitud a nivel de par (se acepta confundir una seña con su pareja);
       - porcentaje de errores que caen dentro del par;
       - exactitud al distinguir las dos señas de cada par (azar 50 %);
       - exactitud entre señas NO parecidas sin reentrenar: promedio de los 32
         subconjuntos de 5 señas con una de cada par (azar 20 %), y los grupos
         A y B por separado;
       - modelos entrenados solo con señas no parecidas (grupos A y B).
     (tabla_parecidas_*.tex, parecidas_*.csv, figuras/).
  3. resumen_semillas.md con todo lo anterior.

Uso:  python analisis_semillas.py [--metricas metricas] [--out metricas/analisis_semillas] [--decimal-punto]
"""
import argparse
import csv
import glob
import itertools
import json
import os
import re

import numpy as np

import evaluate_metrics as em
from evaluate_metrics import fmt_pct, latex_table, plt, save_fig, tex_escape, write_csv

PARES = [("marzo", "rojo"), ("sabado", "agosto"), ("jugar", "broma"), ("nombre", "martes"), ("miercoles", "soltero")]
GRUPO_A = [a for a, _ in PARES]
GRUPO_B = [b for _, b in PARES]
ORDEN_PARES = [c for par in PARES for c in par]

NOMBRES = {
    "features2048_ligera": "Inception V3 (2048), LSTM ligera",
    "probs_ligera": "Inception V3 reentrenada (probabilidades), LSTM ligera",
    "landmarks_ligera": "Coordenadas de MediaPipe, LSTM ligera",
    "features2048_original": "Inception V3 (2048), LSTM original",
    "disimiles_A_features2048": "Solo señas no parecidas, grupo A (5 clases)",
    "disimiles_B_features2048": "Solo señas no parecidas, grupo B (5 clases)",
    "cv_features2048": "Inception V3 (2048)",
    "cv_landmarks": "Coordenadas de MediaPipe",
    "cv_disimiles_A_features2048": "Inception V3 (2048), solo grupo A (5 clases)",
    "cv_disimiles_B_features2048": "Inception V3 (2048), solo grupo B (5 clases)",
}


def seed_dirs(base):
    """{experimento: {semilla: carpeta}}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(base, "*", "s*"))):
        m = re.fullmatch(r"s(\d+)", os.path.basename(d))
        if m and os.path.isfile(os.path.join(d, "resumen.json")):
            out.setdefault(os.path.basename(os.path.dirname(d)), {})[int(m.group(1))] = d
    return out


def mean_sd(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pm(values, coma, latex=True):
    m, s = mean_sd(values)
    sep = r" $\pm$ " if latex else " ± "
    return fmt_pct(m, coma=coma, latex=latex) + sep + fmt_pct(s, coma=coma, latex=latex)


# ---------------------------------------------------------------------------
# Probabilidades por video
# ---------------------------------------------------------------------------
def read_probs(path):
    """-> (clases, y_real (índices), proba (n, K))"""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = [k[2:] for k in rows[0] if k.startswith("p_")]
    y = np.array([classes.index(r["clase_real"]) for r in rows])
    proba = np.array([[float(r["p_" + c]) for c in classes] for r in rows])
    return classes, y, proba


def restricted_accuracy(classes, y, proba, subset):
    """Exactitud sobre los videos de `subset`, eligiendo solo entre las señas de `subset`."""
    idx = [classes.index(c) for c in subset]
    m = np.isin(y, idx)
    if not m.any():
        return None, 0
    pred = np.array(idx)[proba[m][:, idx].argmax(1)]
    return float((pred == y[m]).mean()), int(m.sum())


def pair_analysis(classes, y, proba):
    """Métricas de señas parecidas para un archivo de probabilidades de 10 clases."""
    pair_of = {c: i for i, par in enumerate(PARES) for c in par}
    pred = proba.argmax(1)
    yp = np.array([pair_of[classes[i]] for i in y])
    pp = np.array([pair_of[classes[i]] for i in pred])
    err = pred != y
    res = {
        "exactitud_10": float((~err).mean()),
        "exactitud_par": float((yp == pp).mean()),
        "errores": int(err.sum()),
        "errores_dentro_del_par": int((err & (yp == pp)).sum()),
        "por_par": {},
    }
    res["pct_errores_dentro_del_par"] = res["errores_dentro_del_par"] / res["errores"] if res["errores"] else None
    bins = []
    for a, b in PARES:
        acc, n = restricted_accuracy(classes, y, proba, [a, b])
        ia, ib = classes.index(a), classes.index(b)
        conf = int(((y == ia) & (pred == ib)).sum() + ((y == ib) & (pred == ia)).sum())
        res["por_par"][f"{a}-{b}"] = {"binaria": acc, "n": n, "confusiones_10": conf}
        bins.append(acc)
    res["exactitud_binaria_dentro_del_par"] = float(np.mean(bins))
    subsets = list(itertools.product(*PARES))  # 2^5 = 32 subconjuntos de señas no parecidas
    res["exactitud_no_parecidas_32"] = float(np.mean([restricted_accuracy(classes, y, proba, s)[0] for s in subsets]))
    res["exactitud_grupo_A"] = restricted_accuracy(classes, y, proba, GRUPO_A)[0]
    res["exactitud_grupo_B"] = restricted_accuracy(classes, y, proba, GRUPO_B)[0]
    cm = np.zeros((len(ORDEN_PARES),) * 2, dtype=int)
    for t, p in zip(y, pred):
        cm[ORDEN_PARES.index(classes[t]), ORDEN_PARES.index(classes[p])] += 1
    res["_cm"] = cm
    return res


def prob_file(d):
    for name in ("probabilidades_test.csv", "probabilidades_validacion_cruzada.csv"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_pair_confusion(cm, title, out, name):
    cm = np.asarray(cm, dtype=float)
    data = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    K = len(ORDEN_PARES)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(K), ORDEN_PARES, rotation=45, ha="right")
    ax.set_yticks(range(K), ORDEN_PARES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    for i in range(K):
        for j in range(K):
            if data[i, j] > 0:
                ax.text(j, i, f"{100 * data[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > 0.55 else em.INK)
    for k in range(len(PARES)):  # bloques 2 x 2 de cada par de señas parecidas
        ax.add_patch(plt.Rectangle((2 * k - 0.5, 2 * k - 0.5), 2, 2, fill=False, edgecolor=em.INK, linewidth=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    cb.set_label("% de la fila (exhaustividad)")
    return save_fig(fig, out, name)


def plot_seeds(series, title, xlabel, out, name, chance=None, coma=True):
    """Un punto por semilla y la media ± desvío por experimento (una sola serie)."""
    labels = list(series)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.4))
    for k, lab in enumerate(labels):
        v = np.asarray(series[lab], dtype=float)
        ax.scatter(100 * v, np.full(len(v), k), s=36, color=em.SERIES[0], alpha=0.55, zorder=3,
                   edgecolors="white", linewidths=1.5)
        m, s = mean_sd(v)
        ax.errorbar(100 * m, k, xerr=100 * s, fmt="D", color=em.INK, markersize=6, capsize=4, linewidth=1.5, zorder=4)
        ax.annotate(f"{em.fmt_num(100 * m, 1, coma)} ± {em.fmt_num(100 * s, 1, coma)}", (100 * m, k), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=em.INK2)
    if chance is not None:
        ax.axvline(100 * chance, color=em.INK2, linestyle="--", linewidth=1)
        ax.text(100 * chance + 0.8, len(labels) - 0.45, "azar", color=em.INK2, fontsize=8, ha="left")
    top = max(100 * max(map(max, series.values())), 100 * (chance or 0))
    ax.set_xlim(0, min(100, top + 10))
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-0.6, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out, name)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metricas", default="metricas")
    p.add_argument("--out", default=os.path.join("metricas", "analisis_semillas"))
    p.add_argument("--decimal-punto", action="store_true")
    args = p.parse_args()
    coma = not args.decimal_punto
    out = args.out
    os.makedirs(os.path.join(out, "tablas"), exist_ok=True)
    md = ["# Experimentos con varias semillas y señas parecidas", "",
          "Pares de señas parecidas: " + ", ".join(f"({a}, {b})" for a, b in PARES) + ".", ""]

    # --- 1. Test: media ± desvío entre semillas
    test = seed_dirs(os.path.join(args.metricas, "semillas"))
    rows_csv, rows_tex, md_rows, fig_series = [], [], [], {}
    for exp, seeds in test.items():
        accs, f1s, kappas, hits = [], [], [], []
        for s, d in sorted(seeds.items()):
            e = json.load(open(os.path.join(d, "resumen.json"), encoding="utf-8"))["evaluacion"]["test"]
            accs.append(e["exactitud"]); f1s.append(e["f1_macro"]); kappas.append(e["kappa_cohen"]); hits.append(e["aciertos"])
            rows_csv.append([exp, s, e["n"], e["K"], e["aciertos"], e["exactitud"], e["f1_macro"], e["kappa_cohen"]])
        n = json.load(open(os.path.join(next(iter(seeds.values())), "resumen.json")))["evaluacion"]["test"]
        name = NOMBRES.get(exp, exp)
        rows_tex.append([tex_escape(name), len(seeds), n["K"], pm(accs, coma), pm(f1s, coma),
                         em.fmt_num(mean_sd(kappas)[0], 2, coma), f"{min(hits)} -- {max(hits)} / {n['n']}"])
        md_rows.append(f"| {name} | {len(seeds)} | {n['K']} | {pm(accs, coma, False)} | {pm(f1s, coma, False)} | "
                       f"{em.fmt_num(mean_sd(kappas)[0], 2, coma)} | {min(hits)} – {max(hits)} de {n['n']} |")
        if n["K"] == 10:
            fig_series[name] = accs
    if rows_csv:
        write_csv(os.path.join(out, "semillas_test.csv"),
                  ["experimento", "semilla", "videos", "clases", "aciertos", "exactitud", "f1_macro", "kappa"], rows_csv)
        latex_table(os.path.join(out, "tablas", "tabla_semillas_test.tex"),
                    "Conjunto de prueba: media $\\pm$ desvío estándar entre semillas (42 a 46).", "tab:semillas-test",
                    ["Modelo", "Semillas", "Clases", "Exactitud", "$F_1$ macro", r"$\kappa$", "Aciertos"],
                    rows_tex, align="lrrrrrr",
                    nota="La separación train/validación es la misma en todas las semillas; cambian la "
                         "inicialización de los pesos y el orden de los lotes. Azar: 10 \\% con 10 clases, 20 \\% con 5.")
        md += ["## Conjunto de prueba (20 videos)", "",
               "| Modelo | Semillas | Clases | Exactitud | F1 macro | κ medio | Aciertos (mín. – máx.) |",
               "|---|---:|---:|---|---|---:|---|"] + md_rows + [""]
        if fig_series:
            plot_seeds(fig_series, "Exactitud en prueba por semilla", "Exactitud en prueba (%)", out,
                       "semillas_exactitud_test", chance=0.1, coma=coma)

    # --- 2. Validación cruzada por persona
    cvs = seed_dirs(os.path.join(args.metricas, "semillas_cv"))
    rows_csv, rows_tex, md_rows, fig_series = [], [], [], {}
    for exp, seeds in cvs.items():
        accs, f1s, folds_all = [], [], []
        for s, d in sorted(seeds.items()):
            c = json.load(open(os.path.join(d, "resumen.json"), encoding="utf-8"))["validacion_cruzada"]
            accs.append(c["exactitud_media"]); f1s.append(c["f1_macro_media"])
            folds_all += [f["exactitud"] for f in c["pliegues"]]
            rows_csv.append([exp, s, c["k"], c["exactitud_media"], c["exactitud_desvio"], c["f1_macro_media"]])
        K = len(read_probs(prob_file(next(iter(seeds.values()))))[0]) if prob_file(next(iter(seeds.values()))) else ""
        name = NOMBRES.get(exp, exp)
        rows_tex.append([tex_escape(name), len(seeds), K, pm(accs, coma), pm(f1s, coma),
                         fmt_pct(mean_sd(folds_all)[1], coma=coma)])
        md_rows.append(f"| {name} | {len(seeds)} | {K} | {pm(accs, coma, False)} | {pm(f1s, coma, False)} | "
                       f"{fmt_pct(mean_sd(folds_all)[1], coma=coma, latex=False)} |")
        if K == 10:
            fig_series[name] = accs
    if rows_csv:
        write_csv(os.path.join(out, "semillas_cv.csv"),
                  ["experimento", "semilla", "pliegues", "exactitud_media", "exactitud_desvio_pliegues", "f1_macro_media"],
                  rows_csv)
        latex_table(os.path.join(out, "tablas", "tabla_semillas_cv.tex"),
                    "Validación cruzada de 5 pliegues agrupada por persona: media $\\pm$ desvío estándar entre semillas.",
                    "tab:semillas-cv",
                    ["Representación", "Semillas", "Clases", "Exactitud", "$F_1$ macro", "Desvío entre pliegues"],
                    rows_tex, align="lrrrrr",
                    nota="Cada semilla cambia la asignación de personas a pliegues y la inicialización. "
                         "Se evalúa solo sobre videos originales.")
        md += ["## Validación cruzada por persona (100 videos originales)", "",
               "| Representación | Semillas | Clases | Exactitud | F1 macro | Desvío entre pliegues |",
               "|---|---:|---:|---|---|---:|"] + md_rows + [""]
        if fig_series:
            plot_seeds(fig_series, "Validación cruzada por persona, por semilla", "Exactitud media de los 5 pliegues (%)",
                       out, "semillas_exactitud_cv", chance=0.1, coma=coma)

    # --- 3. Señas parecidas
    md += ["## Señas parecidas", ""]
    for origen, groups in (("prueba", test), ("validación cruzada por persona", cvs)):
        rows_csv, rows_tex, md_rows, pair_rows = [], [], [], []
        five = []
        for exp, seeds in groups.items():
            per_seed = []
            for s, d in sorted(seeds.items()):
                pf = prob_file(d)
                if pf is None:
                    continue
                classes, y, proba = read_probs(pf)
                if len(classes) == 10:
                    r = pair_analysis(classes, y, proba)
                    per_seed.append(r)
                    rows_csv.append([exp, s, r["exactitud_10"], r["exactitud_par"], r["pct_errores_dentro_del_par"],
                                     r["exactitud_binaria_dentro_del_par"], r["exactitud_no_parecidas_32"],
                                     r["exactitud_grupo_A"], r["exactitud_grupo_B"]])
                elif set(classes) in (set(GRUPO_A), set(GRUPO_B)):
                    five.append((exp, s, float((proba.argmax(1) == y).mean())))
            if not per_seed:
                continue
            name = NOMBRES.get(exp, exp)
            get = lambda k: [r[k] for r in per_seed]  # noqa: E731
            vals = [pm(get("exactitud_10"), coma), pm(get("exactitud_par"), coma),
                    pm(get("pct_errores_dentro_del_par"), coma), pm(get("exactitud_binaria_dentro_del_par"), coma),
                    pm(get("exactitud_no_parecidas_32"), coma)]
            rows_tex.append([tex_escape(name)] + vals)
            md_rows.append(f"| {name} | " + " | ".join(v.replace(r" $\pm$ ", " ± ").replace(r"\,\%", " %")
                                                         for v in vals) + " |")
            for a, b in PARES:
                k = f"{a}-{b}"
                pair_rows.append([exp, k, mean_sd([r["por_par"][k]["binaria"] for r in per_seed])[0],
                                  float(np.mean([r["por_par"][k]["confusiones_10"] for r in per_seed])),
                                  per_seed[0]["por_par"][k]["n"]])
            cm = sum(r["_cm"] for r in per_seed)
            slug = "test" if origen == "prueba" else "cv"
            plot_pair_confusion(cm, f"{name} — {origen}\n(suma de {len(per_seed)} semilla{'s' if len(per_seed) > 1 else ''})", out,
                                f"confusion_pares_{slug}_{exp}")
        if not rows_csv:
            continue
        slug = "test" if origen == "prueba" else "cv"
        write_csv(os.path.join(out, f"parecidas_{slug}.csv"),
                  ["experimento", "semilla", "exactitud_10", "exactitud_par", "pct_errores_dentro_del_par",
                   "exactitud_binaria_dentro_del_par", "exactitud_no_parecidas_32", "exactitud_grupo_A",
                   "exactitud_grupo_B"], rows_csv)
        write_csv(os.path.join(out, f"parecidas_por_par_{slug}.csv"),
                  ["experimento", "par", "exactitud_binaria_media", "confusiones_dentro_del_par_media", "videos"],
                  pair_rows)
        latex_table(os.path.join(out, "tablas", f"tabla_parecidas_{slug}.tex"),
                    f"Señas parecidas ({origen}): media $\\pm$ desvío entre semillas.", f"tab:parecidas-{slug}",
                    ["Modelo", "10 señas", "Par correcto", "Errores dentro del par", "Dentro del par (2 señas)",
                     "No parecidas (5 señas)"], rows_tex, align="lrrrrr",
                    nota="Par correcto: se acepta confundir una seña con su pareja (5 grupos, azar 20 \\%). "
                         "Dentro del par: elegir entre las dos señas del par (azar 50 \\%). No parecidas: promedio de "
                         "los 32 subconjuntos de 5 señas con una de cada par, eligiendo solo entre esas 5 (azar 20 \\%); "
                         "el modelo es el mismo de 10 señas, sin reentrenar.")
        md += [f"### {origen.capitalize()}", "",
               "| Modelo | 10 señas | Par correcto (azar 20 %) | Errores dentro del par | "
               "Dentro del par, 2 señas (azar 50 %) | No parecidas, 5 señas (azar 20 %) |",
               "|---|---|---|---|---|---|"] + md_rows + [""]
        md += ["Por par (media entre semillas):", "", "| Modelo | Par | Distinguir las dos señas | Confusiones dentro del par (de 10 señas) |",
               "|---|---|---:|---:|"]
        md += [f"| {NOMBRES.get(e, e)} | {k} | {fmt_pct(a, coma=coma, latex=False)} | {em.fmt_num(c, 1, coma)} de {n} videos |"
               for e, k, a, c, n in pair_rows] + [""]

        # Modelos entrenados solo con señas no parecidas vs. los de 10 señas restringidos al mismo grupo
        if five:
            base = "features2048_ligera" if origen == "prueba" else "cv_features2048"
            ref = {g: [r[7 if g == "A" else 8] for r in rows_csv if r[0] == base] for g in ("A", "B")}
            md += ["Señas no parecidas: modelo de 10 señas restringido al grupo vs. modelo entrenado solo con ese grupo "
                   f"({origen}, Inception V3 2048, LSTM ligera; azar 20 %):", "",
                   "| Grupo | Señas | 10 señas, restringido | Entrenado solo con el grupo |", "|---|---|---|---|"]
            tex5 = []
            for g, grupo in (("A", GRUPO_A), ("B", GRUPO_B)):
                trained = [a for e, s, a in five if f"disimiles_{g}_" in e]
                if not trained:
                    continue
                md.append(f"| {g} | {', '.join(grupo)} | {pm(ref[g], coma, False)} | {pm(trained, coma, False)} |")
                tex5.append([g, ", ".join(grupo), pm(ref[g], coma), pm(trained, coma)])
            md.append("")
            latex_table(os.path.join(out, "tablas", f"tabla_no_parecidas_{slug}.tex"),
                        f"Señas no parecidas ({origen}): exactitud con 5 señas, una de cada par.",
                        f"tab:no-parecidas-{slug}",
                        ["Grupo", "Señas", "Modelo de 10 señas, restringido", "Modelo entrenado con el grupo"],
                        tex5, align="llrr", nota="Media $\\pm$ desvío entre semillas. Azar: 20 \\%.")

    with open(os.path.join(out, "resumen_semillas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSalida en {out}/")


if __name__ == "__main__":
    main()
