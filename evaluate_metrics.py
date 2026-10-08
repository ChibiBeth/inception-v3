#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_metrics.py — Métricas completas para el libro de tesis (LSPy).

Genera, a partir de los artefactos que ya produce el pipeline
(data/data_file.csv, data/train|test/<Clase>/*.jpg, data/sequences/*.npy,
data/logs/*.log, el modelo LSTM entrenado), un informe con:

  1. Auditoría del conjunto de datos (sin TensorFlow):
       - videos por clase y partición, originales vs. aumentados
       - cobertura de detección de manos por video (fotogramas reales vs. relleno)
       - detección de "colisiones de nombre" (secuencias que mezclan fotogramas
         de varios videos por el glob de extract_features_harp.py)
       - control de fuga de datos: videos de prueba cuyo original aparece en train
       - estadísticas de los vectores de características (.npy)
       - (opcional) fps, resolución y duración de los videos crudos
  2. Curvas de entrenamiento a partir de los .log de CSVLogger.
  3. Evaluación del modelo (por partición):
       exactitud, exactitud balanceada, top-k, precisión / exhaustividad / F1
       por clase y promedios macro/ponderado, matriz de confusión (conteos y
       normalizada), kappa de Cohen, MCC, log-loss, Brier, ROC-AUC y curvas
       ROC/PR uno-contra-resto, calibración (ECE + diagrama de confiabilidad),
       intervalos de confianza bootstrap al 95 %, comparación con el azar
       (1/K, clase mayoritaria, prueba binomial), pares de clases más
       confundidas, predicciones por video, parámetros, tamaño y latencia.
  4. (Opcional) Proyección 2D (PCA / t-SNE) de la representación interna de la LSTM.
  5. (Opcional) Validación cruzada estratificada y agrupada por video original
     (o por persona), entrenando de cero la misma arquitectura en cada pliegue.

Todo se guarda en --out-dir como:
  resumen.json, resumen.md, *.csv, tablas LaTeX (booktabs) y figuras PDF + PNG.

Uso típico (desde la raíz del repositorio, con el venv activado):

  # Solo auditoría de datos + curvas (no requiere modelo):
  python evaluate_metrics.py --sin-modelo

  # Evaluación completa del modelo guardado por train_lstm_harp.py:
  python evaluate_metrics.py --model lstm_senha_model --splits test train

  # Usar el mejor checkpoint (menor val_loss) en data/checkpoints:
  python evaluate_metrics.py --model mejor

  # Agregar embeddings, latencia de Inception y validación cruzada de 5 pliegues:
  python evaluate_metrics.py --model lstm_senha_model --embeddings \
      --latencia-inception --cv 5 --cv-epochs 60 --raw-dir rawdata_clean

Requisitos extra respecto de requirements.txt:  scikit-learn  (pip install scikit-learn==1.5.2)
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Estilo de figuras (sobrio, apto para impresión)
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "lines.linewidth": 2,
})

from lspy_common import (TRIM, frame_pattern, is_augmented_row, load_meta,  # noqa: E402
                         original_stem, sequence_file, video_original)


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)


def save_fig(fig, out_dir, name):
    os.makedirs(os.path.join(out_dir, "figuras"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "figuras", f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join("figuras", f"{name}.pdf")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_num(x, dec=3, coma=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:.{dec}f}"
    return s.replace(".", ",") if coma else s


def fmt_pct(x, dec=1, coma=True, latex=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{100 * x:.{dec}f}"
    if coma:
        s = s.replace(".", ",")
    return s + (r"\,\%" if latex else " %")


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


def latex_table(path, caption, label, header, rows, align=None, nota=None):
    """Escribe una tabla booktabs lista para \\input{} en la tesis."""
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [
        "% Generado automáticamente por evaluate_metrics.py — " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "% Requiere \\usepackage{booktabs} en el preámbulo.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for r in rows:
        if r == "MIDRULE":
            lines.append(r"    \midrule")
        else:
            lines.append("    " + " & ".join(str(c) for c in r) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if nota:
        lines.append(f"  \\par\\vspace{{2pt}}\\footnotesize {nota}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def source_tag(stem):
    """Etiqueta de origen deducida del nombre (NO necesariamente la persona)."""
    o = original_stem(stem).lower()
    if o.startswith("video_"):
        return "video_<fecha>"
    m = re.match(r"^[a-záéíóúñ]+_(.+)$", o)
    return m.group(1) if m else o


def load_signers_csv(path):
    """CSV opcional con columnas: video_original,persona[,particion,clase].

    El nombre puede llevar extensión. Si está la columna particion, la persona se busca
    por (partición, video): así un original de test como 'marzo_cr1_2' no se confunde
    con la copia aumentada 'marzo_cr1_2' de train."""
    if not path:
        return None
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if (len(row) >= 2 and row[0].strip() and row[1].strip() and not row[0].startswith("#")
                    and row[0].strip() != "video_original"):
                stem = TRIM.sub("", os.path.splitext(row[0].strip())[0]).strip()
                split = row[2].strip() if len(row) >= 3 and row[2].strip() else None
                mapping[(split, stem)] = row[1].strip()
    return mapping


def signer_of(signers, r):
    orig = video_original(r["split"], r["stem"])
    return signers.get((r["split"], orig), signers.get((None, orig), "desconocido"))


# ---------------------------------------------------------------------------
# Carga del conjunto de datos (misma lógica de filtrado que DataSet)
# ---------------------------------------------------------------------------
def read_data_file(data_dir):
    path = os.path.join(data_dir, "data_file.csv")
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] No se encontró {path}. ¿Se ejecutó handtrack.py?")
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 4]
    # handtrack.py actual agrega una 5.ª columna: fotogramas con mano en la secuencia
    return [{"split": r[0], "clase": r[1], "stem": r[2], "nb_frames": int(r[3]),
             "con_mano": int(r[4]) if len(r) >= 5 and r[4].isdigit() else None} for r in rows]


def filter_rows(rows, seq_length, max_frames, class_limit, only=None):
    classes = sorted({r["clase"] for r in rows})
    if only:
        classes = sorted(only)
    elif class_limit:
        classes = classes[:class_limit]
    kept = [r for r in rows if seq_length <= r["nb_frames"] <= max_frames and r["clase"] in classes]
    return kept, classes


def seq_path(data_dir, r, seq_length, data_type):
    return sequence_file(data_dir, r["split"], r["stem"], seq_length, data_type)


def load_sequences(data_dir, rows, classes, seq_length, data_type):
    X, y, used = [], [], []
    missing = 0
    for r in rows:
        p = seq_path(data_dir, r, seq_length, data_type)
        if not os.path.isfile(p):
            missing += 1
            continue
        X.append(np.load(p))
        y.append(classes.index(r["clase"]))
        used.append(r)
    if missing:
        log(f"  [aviso] {missing} secuencias .npy no encontradas (se omiten).")
    if not X:
        return None, None, []
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), used


# ---------------------------------------------------------------------------
# 1. Auditoría del conjunto de datos
# ---------------------------------------------------------------------------
def audit_dataset(args, rows, classes, signers):
    out = args.out_dir
    res = {}
    log("\n=== 1. Auditoría del conjunto de datos ===")

    # --- 1.1 Conteos por clase / partición / original-aumentado
    splits = sorted({r["split"] for r in rows})
    cnt = defaultdict(Counter)
    for r in rows:
        key = r["split"] + ("_aug" if is_augmented_row(r["split"], r["stem"]) else "_orig")
        cnt[r["clase"]][key] += 1
    header = ["Clase"]
    for s in splits:
        header += [f"{s} orig.", f"{s} aum.", f"{s} total"]
    table, tex_rows = [], []
    totals = Counter()
    for c in classes:
        row = [c]
        for s in splits:
            o, a = cnt[c][s + "_orig"], cnt[c][s + "_aug"]
            row += [o, a, o + a]
            totals[s + "_orig"] += o
            totals[s + "_aug"] += a
        table.append(row)
        tex_rows.append([tex_escape(row[0])] + row[1:])
    tot_row = ["Total"]
    for s in splits:
        tot_row += [totals[s + "_orig"], totals[s + "_aug"], totals[s + "_orig"] + totals[s + "_aug"]]
    write_csv(os.path.join(out, "dataset_conteos.csv"), header, table + [tot_row])
    latex_table(os.path.join(out, "tablas", "tabla_dataset.tex"),
                "Cantidad de videos por clase, partición y tipo (original o aumentado).",
                "tab:dataset-conteos", [tex_escape(h) for h in header],
                tex_rows + ["MIDRULE", [r"\textbf{Total}"] + tot_row[1:]],
                nota="Aum.: copias generadas con \\texttt{data\\_augmentation.py}.")
    res["conteos"] = {"clases": classes, "por_clase": {c: dict(cnt[c]) for c in classes},
                      "totales": dict(totals)}
    log(f"  Clases ({len(classes)}): {', '.join(classes)}")
    for s in splits:
        log(f"  {s}: {totals[s + '_orig']} originales + {totals[s + '_aug']} aumentados")

    # --- 1.2 Fuga de datos / independencia de personas
    orig_by_split = defaultdict(set)
    for r in rows:
        orig_by_split[r["split"]].add((r["clase"], video_original(r["split"], r["stem"])))
    leak = sorted(orig_by_split.get("train", set()) & orig_by_split.get("test", set()))
    res["fuga_mismo_video_original_en_train_y_test"] = [f"{c}/{s}" for c, s in leak]
    if leak:
        log(f"  [ALERTA] {len(leak)} videos originales aparecen en train y test: {leak[:5]}...")
    else:
        log("  Fuga por nombre de video original: ninguna (train y test no comparten videos originales).")

    group_fn = (lambda r: signer_of(signers, r)) if signers else (lambda r: source_tag(r["stem"]))
    group_name = "persona" if signers else "etiqueta de origen (deducida del nombre)"
    tags = defaultdict(Counter)
    for r in rows:
        tags[r["split"]][group_fn(r)] += 1
    shared = sorted(set(tags.get("train", {})) & set(tags.get("test", {})))
    res["grupos"] = {"tipo": group_name, "por_particion": {k: dict(v) for k, v in tags.items()},
                     "compartidos_train_test": shared}
    log(f"  Grupos por {group_name}: " + "; ".join(f"{k}={dict(v)}" for k, v in tags.items()))
    if shared:
        log(f"  [nota] Grupos presentes en train y test: {shared}. "
            "Si corresponden a las mismas personas, la evaluación NO es independiente del signante.")

    # --- 1.3 Cobertura de manos, relleno y colisiones de glob
    cov_rows, collisions = [], []
    have_frames = False
    for r in rows:
        folder = os.path.join(args.data_dir, r["split"], r["clase"])
        if not os.path.isdir(folder):
            continue
        st = glob.escape(r["stem"])
        real = len(glob.glob(os.path.join(folder, st + "-[0-9][0-9][0-9][0-9].jpg")))
        pad = len(glob.glob(os.path.join(folder, st + "_[0-9][0-9][0-9][0-9].jpg")))
        matched = len(glob.glob(os.path.join(folder, frame_pattern(r["stem"]))))  # = lo que usa extract_features
        if real + pad == 0:
            continue
        if r["con_mano"] is not None:  # sin relleno: los cuadros sin mano quedan en negro
            real, pad = r["con_mano"], real + pad - r["con_mano"]
        have_frames = True
        extra = matched - (real + pad)
        cov_rows.append([r["split"], r["clase"], r["stem"], real, pad, matched, extra,
                         round(real / max(1, args.seq_length), 4)])
        if extra > 0:
            collisions.append(f"{r['split']}/{r['clase']}/{r['stem']} (+{extra} fotogramas ajenos)")
    if have_frames:
        write_csv(os.path.join(out, "cobertura_manos_por_video.csv"),
                  ["particion", "clase", "video", "fotogramas_con_mano", "fotogramas_relleno",
                   "fotogramas_que_toma_el_glob", "fotogramas_ajenos", "proporcion_real"], cov_rows)
        res["colisiones_glob"] = {"cantidad": len(collisions), "ejemplos": collisions[:30]}
        if collisions:
            log(f"  [ALERTA] {len(collisions)} secuencias toman fotogramas de OTROS videos "
                f"(patrón de glob de extract_features_harp.py). Ej.: {collisions[:3]}")
        else:
            log("  Colisiones de nombre en el glob: ninguna.")

        # tabla de cobertura por clase
        by = defaultdict(list)
        for row in cov_rows:
            by[(row[0], row[1])].append(row[3])
        cov_tab, cov_tex = [], []
        for s in splits:
            for c in classes:
                v = by.get((s, c))
                if not v:
                    continue
                v = np.array(v)
                cov_tab.append([s, c, len(v), float(v.mean()), float(v.min()), float(v.max()),
                                float((v / args.seq_length).mean())])
                cov_tex.append([s, tex_escape(c), len(v), fmt_num(v.mean(), 1), int(v.min()), int(v.max()),
                                fmt_pct((v / args.seq_length).mean())])
        write_csv(os.path.join(out, "cobertura_manos_por_clase.csv"),
                  ["particion", "clase", "videos", "media_fotogramas_con_mano", "min", "max",
                   "proporcion_media_real"], cov_tab)
        latex_table(os.path.join(out, "tablas", "tabla_cobertura_manos.tex"),
                    f"Fotogramas con al menos una mano detectada por MediaPipe (sobre {args.seq_length}).",
                    "tab:cobertura-manos",
                    ["Partición", "Clase", "Videos", "Media", "Mín.", "Máx.", r"\% real"],
                    cov_tex, align="llrrrrr",
                    nota=("El resto de la secuencia son fotogramas sin manos (en negro), conservados "
                          "al remuestrear el video por tiempo."
                          if any(r["con_mano"] is not None for r in rows) else
                          "El resto de la secuencia se completa con copias del último fotograma (relleno)."))
        # figura: distribución de fotogramas reales por clase
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        data = [[row[3] for row in cov_rows if row[1] == c] for c in classes]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.5))
        ax.set_xticks(range(1, len(classes) + 1), classes)
        for b in bp["boxes"]:
            b.set_facecolor(SERIES[0] + "55")
            b.set_edgecolor(SERIES[0])
        ax.axhline(args.seq_length, color=INK2, linestyle="--", linewidth=1)
        ax.text(len(classes) + 0.4, args.seq_length, f"{args.seq_length}", va="center", color=INK2, fontsize=8)
        ax.set_ylabel("Fotogramas con mano detectada")
        ax.set_title("Cobertura de detección de manos por clase")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        res["fig_cobertura"] = save_fig(fig, out, "cobertura_manos_por_clase")
        allv = np.array([row[3] for row in cov_rows])
        res["cobertura_global"] = {"media": float(allv.mean()), "mediana": float(np.median(allv)),
                                   "min": int(allv.min()), "max": int(allv.max()),
                                   "videos_con_menos_de_30": int((allv < 30).sum())}
        log(f"  Fotogramas con mano por video: media {allv.mean():.1f}, mediana {np.median(allv):.0f}, "
            f"mín {allv.min()}, máx {allv.max()}")
    else:
        log("  (No se encontraron fotogramas .jpg en data/<split>/<clase>; se omite cobertura y colisiones.)")

    # --- 1.4 Estadísticas de características
    X, y, used = load_sequences(args.data_dir, rows, classes, args.seq_length, args.data_type)
    if X is not None:
        feat = {"n_secuencias": int(len(X)), "forma": list(X.shape[1:])}
        flat = X.reshape(-1, X.shape[-1])
        sums = flat.sum(1)
        feat["parecen_probabilidades_softmax"] = bool(np.all(flat >= -1e-6) and np.allclose(sums, 1, atol=1e-3))
        feat["media_global"] = float(flat.mean())
        feat["desvio_por_dimension_min"] = float(flat.std(0).min())
        feat["desvio_por_dimension_max"] = float(flat.std(0).max())
        diffs = np.abs(np.diff(X, axis=1)).max(-1)
        feat["proporcion_pasos_identicos"] = float((diffs < 1e-7).mean())
        argmax_frames = flat.argmax(1)
        feat["distribucion_argmax_por_fotograma"] = {str(k): int(v) for k, v in Counter(argmax_frames).items()}
        res["caracteristicas"] = feat
        log(f"  Características: {len(X)} secuencias de forma {X.shape[1:]}; "
            f"¿vectores softmax? {feat['parecen_probabilidades_softmax']}; "
            f"pasos idénticos consecutivos (relleno): {100 * feat['proporcion_pasos_identicos']:.1f} %")
        if feat["parecen_probabilidades_softmax"] and args.data_type == "features":
            log("  [ALERTA] Las 'características' son salidas softmax de una capa Dense NO entrenada "
                f"({X.shape[-1]} dimensiones). Ver docs/05_hallazgos_y_recomendaciones.md (H1).")
    else:
        log("  (No hay secuencias .npy; ¿se ejecutó extract_features_harp.py?)")

    # --- 1.5 Videos crudos (opcional)
    if args.raw_dir:
        try:
            import cv2
        except ImportError:
            cv2 = None
            log("  [aviso] OpenCV no disponible: se omite el análisis de videos crudos.")
        if cv2 is not None:
            vrows = []
            for vp in sorted(glob.glob(os.path.join(args.raw_dir, "*", "*", "*"))):
                if not os.path.isfile(vp):
                    continue
                parts = os.path.normpath(vp).split(os.sep)
                cap = cv2.VideoCapture(vp)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                vrows.append([parts[-3], parts[-2], parts[-1], round(fps, 2), n, w, h,
                              round(n / fps, 3) if fps else "", is_augmented_row(parts[-3], os.path.splitext(parts[-1])[0])])
            write_csv(os.path.join(out, "videos_crudos.csv"),
                      ["particion", "clase", "archivo", "fps", "fotogramas", "ancho", "alto", "duracion_s",
                       "aumentado"], vrows)
            if vrows:
                durs = np.array([r[7] for r in vrows if r[7] != ""], dtype=float)
                res["videos_crudos"] = {
                    "n": len(vrows),
                    "fps": dict(Counter(int(round(r[3])) for r in vrows)),
                    "resoluciones": dict(Counter(f"{r[5]}x{r[6]}" for r in vrows).most_common(10)),
                    "duracion_s": {"media": float(durs.mean()), "min": float(durs.min()), "max": float(durs.max())},
                }
                log(f"  Videos crudos: {len(vrows)}; fps {res['videos_crudos']['fps']}; "
                    f"resoluciones {res['videos_crudos']['resoluciones']}; "
                    f"duración media {durs.mean():.2f} s")
    return res


# ---------------------------------------------------------------------------
# 2. Curvas de entrenamiento (CSVLogger)
# ---------------------------------------------------------------------------
def read_training_log(path):
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        data = defaultdict(list)
        for row in rd:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (TypeError, ValueError):
                    pass
    return data


def training_curves(args):
    log("\n=== 2. Curvas de entrenamiento ===")
    if args.log_file:
        files = [args.log_file]
    else:
        # el log del mismo tipo de modelo (train_lstm_harp.py lo nombra lstm-<data_type>-<arch>-training-*)
        meta = None if args.sin_modelo or args.model == "mejor" else load_meta(args.model)
        prefix = meta.get("log", f"lstm-{meta['data_type']}-{meta['arch']}") if meta else None
        pattern = f"{prefix}-training-*.log" if meta else "*.log"
        files = sorted(glob.glob(os.path.join(args.logs_dir, pattern)), key=os.path.getmtime)
        files = files[-1:] if files else []
    if not files:
        log(f"  (No hay .log de CSVLogger en {args.logs_dir}.)")
        return {}
    path = files[0]
    d = read_training_log(path)
    if "epoch" not in d or not d["epoch"]:
        log(f"  (El log {path} está vacío.)")
        return {}
    ep = np.array(d["epoch"]) + 1
    acc_k = "accuracy" if "accuracy" in d else ("acc" if "acc" in d else None)
    vacc_k = "val_accuracy" if "val_accuracy" in d else ("val_acc" if "val_acc" in d else None)
    res = {"archivo": path, "epocas": int(len(ep))}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (tk, vk, title) in zip(axes, [("loss", "val_loss", "Pérdida (entropía cruzada)"),
                                          (acc_k, vacc_k, "Exactitud")]):
        if tk and tk in d:
            ax.plot(ep, d[tk], color=SERIES[0], label="Entrenamiento")
        if vk and vk in d:
            ax.plot(ep, d[vk], color=SERIES[1], label="Validación")
        ax.set_xlabel("Época")
        ax.set_title(title)
    if "val_loss" in d:
        best = int(np.argmin(d["val_loss"]))
        res["mejor_epoca_val_loss"] = int(ep[best])
        res["val_loss_min"] = float(d["val_loss"][best])
        if vacc_k:
            res["val_exactitud_en_mejor_epoca"] = float(d[vacc_k][best])
            res["val_exactitud_max"] = float(np.max(d[vacc_k]))
        for ax in axes:
            ax.axvline(ep[best], color=INK2, linestyle=":", linewidth=1)
        axes[0].annotate(f"mejor época: {int(ep[best])}", (ep[best], d["val_loss"][best]),
                         textcoords="offset points", xytext=(6, 10), fontsize=8, color=INK2)
    if acc_k:
        res["exactitud_entrenamiento_final"] = float(d[acc_k][-1])
        if vacc_k:
            res["brecha_final_train_val"] = float(d[acc_k][-1] - d[vacc_k][-1])
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    res["figura"] = save_fig(fig, args.out_dir, "curvas_entrenamiento")
    log(f"  Log: {path} ({len(ep)} épocas). " +
        (f"Mejor val_loss en época {res.get('mejor_epoca_val_loss')}." if "mejor_epoca_val_loss" in res else ""))
    if "val_loss" in d and os.path.basename(path).startswith("lstm-training-"):  # log de la versión anterior
        log("  [nota] En train_lstm_harp.py la 'validación' es el conjunto de PRUEBA: elegir la mejor "
            "época con él sesga a favor la exactitud reportada.")
    return res


# ---------------------------------------------------------------------------
# 3. Modelo: carga, predicción, latencia
# ---------------------------------------------------------------------------
def resolve_model_path(arg, data_dir):
    if arg != "mejor":
        return arg
    cands = glob.glob(os.path.join(data_dir, "checkpoints", "*.hdf5")) + \
        glob.glob(os.path.join(data_dir, "checkpoints", "*.keras")) + \
        glob.glob(os.path.join(data_dir, "checkpoints", "*.h5"))
    best, best_loss = None, float("inf")
    for c in cands:
        m = re.search(r"-(\d+\.\d+)\.(hdf5|h5|keras)$", c)
        if m and float(m.group(1)) < best_loss:
            best, best_loss = c, float(m.group(1))
    if not best:
        sys.exit("[ERROR] No se encontraron checkpoints en data/checkpoints.")
    log(f"  Mejor checkpoint: {best} (val_loss={best_loss})")
    return best


def load_model_any(path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model, model
    except Exception as e:  # Keras 3 no abre SavedModel con load_model
        if os.path.isdir(path):
            log(f"  [aviso] load_model falló ({type(e).__name__}); se usa TFSMLayer (solo inferencia).")
            layer = tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")
            return layer, None
        raise


def _call(model, x):
    try:
        return model(x, training=False)
    except TypeError:  # p. ej. TFSMLayer no acepta 'training'
        return model(x)


def predict_proba(model, X, batch=16):
    import tensorflow as tf
    outs = []
    for i in range(0, len(X), batch):
        o = _call(model, tf.constant(X[i:i + batch]))
        if isinstance(o, dict):
            o = list(o.values())[0]
        outs.append(np.asarray(o))
    return np.concatenate(outs, 0)


def model_info(keras_model, path):
    info = {}
    if keras_model is not None:
        try:
            info["parametros_totales"] = int(keras_model.count_params())
            info["parametros_entrenables"] = int(sum(int(np.prod(w.shape)) for w in keras_model.trainable_weights))
            info["capas"] = [f"{l.__class__.__name__}({getattr(l, 'units', '')})" for l in keras_model.layers]
        except Exception:
            pass
    size = 0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    elif os.path.isfile(path):
        size = os.path.getsize(path)
    info["tamano_en_disco_MB"] = round(size / 1e6, 2)
    return info


def measure_latency(model, sample, runs=30):
    import tensorflow as tf
    x = tf.constant(sample[None])
    for _ in range(3):
        _call(model, x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _call(model, x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"media_ms": float(ts.mean()), "desvio_ms": float(ts.std()), "p95_ms": float(np.percentile(ts, 95)),
            "corridas": runs}


def inception_latency(runs=20):
    """Tiempo de InceptionV3 por fotograma (299x299) — costo dominante del pipeline."""
    import tensorflow as tf
    base = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
    x = tf.random.uniform((1, 299, 299, 3), -1, 1)
    for _ in range(3):
        base(x, training=False)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        base(x, training=False)
        ts.append((time.perf_counter() - t0) * 1000)
    xb = tf.random.uniform((30, 299, 299, 3), -1, 1)
    base(xb, training=False)
    t0 = time.perf_counter()
    base(xb, training=False)
    batch_ms = (time.perf_counter() - t0) * 1000 / 30
    return {"por_fotograma_lote1_ms": float(np.mean(ts)), "por_fotograma_lote30_ms": float(batch_ms),
            "parametros": int(base.count_params())}


# ---------------------------------------------------------------------------
# 3b. Métricas
# ---------------------------------------------------------------------------
def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            table.append((lo, hi, 0, np.nan, np.nan))
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        table.append((lo, hi, int(m.sum()), float(acc), float(cf)))
    return float(ece), table


def binom_pvalue_greater(k, n, p):
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p, alternative="greater").pvalue)
    except Exception:
        return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def bootstrap_ci(y, pred, K, n_boot, seed):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append((y[idx] == pred[idx]).mean())
        f1s.append(f1_score(y[idx], pred[idx], labels=list(range(K)), average="macro", zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"exactitud_IC95": q(accs), "f1_macro_IC95": q(f1s), "remuestreos": n_boot}


def compute_metrics(y, proba, classes, n_boot=2000, seed=42):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, log_loss, matthews_corrcoef,
                                 precision_recall_fscore_support, roc_auc_score, top_k_accuracy_score)
    K = len(classes)
    labels = list(range(K))
    pred = proba.argmax(1)
    conf = proba.max(1)
    correct = (pred == y).astype(float)
    m = {"n": int(len(y)), "K": K}
    m["exactitud"] = float(accuracy_score(y, pred))
    m["aciertos"] = int(correct.sum())
    m["exactitud_balanceada"] = float(balanced_accuracy_score(y, pred))
    for k in (2, 3, 5):
        if k < K:
            m[f"top{k}"] = float(top_k_accuracy_score(y, proba, k=k, labels=labels))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    m["por_clase"] = {classes[i]: {"precision": float(p[i]), "exhaustividad": float(r[i]),
                                   "f1": float(f[i]), "soporte": int(s[i])} for i in labels}
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        name = "macro" if avg == "macro" else "ponderado"
        m[f"precision_{name}"], m[f"exhaustividad_{name}"], m[f"f1_{name}"] = float(pa), float(ra), float(fa)
    m["kappa_cohen"] = float(cohen_kappa_score(y, pred, labels=labels))
    m["mcc"] = float(matthews_corrcoef(y, pred))
    pc = np.clip(proba, 1e-12, 1)
    pc = pc / pc.sum(1, keepdims=True)
    m["log_loss"] = float(log_loss(y, pc, labels=labels))
    onehot = np.eye(K)[y]
    m["brier"] = float(((proba - onehot) ** 2).sum(1).mean())
    present = np.unique(y)
    if len(present) == K and K > 2:
        try:
            m["roc_auc_macro_ovr"] = float(roc_auc_score(y, pc, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            pass
    m["ece"], m["tabla_calibracion"] = expected_calibration_error(conf, correct)
    m["confianza_media_aciertos"] = float(conf[correct == 1].mean()) if correct.any() else None
    m["confianza_media_errores"] = float(conf[correct == 0].mean()) if (correct == 0).any() else None
    # Líneas base
    m["azar_1_sobre_K"] = 1.0 / K
    maj = Counter(y.tolist()).most_common(1)[0][1]
    m["clase_mayoritaria"] = maj / len(y)
    m["p_valor_binomial_vs_azar"] = binom_pvalue_greater(int(correct.sum()), len(y), 1.0 / K)
    m["resolucion_por_ejemplo"] = 1.0 / len(y)
    m.update(bootstrap_ci(y, pred, K, n_boot, seed) if n_boot > 0 else {})
    cm = confusion_matrix(y, pred, labels=labels)
    m["matriz_confusion"] = cm.tolist()
    pairs = [(classes[i], classes[j], int(cm[i, j])) for i in labels for j in labels if i != j and cm[i, j] > 0]
    m["pares_mas_confundidos"] = sorted(pairs, key=lambda t: -t[2])[:10]
    return m, pred, conf


# ---------------------------------------------------------------------------
# 3c. Figuras de evaluación
# ---------------------------------------------------------------------------
def plot_confusion(cm, classes, title, out_dir, name, normalize=False):
    cm = np.asarray(cm, dtype=float)
    data = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    K = len(classes)
    fig, ax = plt.subplots(figsize=(0.55 * K + 2.6, 0.5 * K + 2.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, data.max()))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(K), classes, rotation=45, ha="right")
    ax.set_yticks(range(K), classes)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    thr = (1 if normalize else data.max()) * 0.55
    for i in range(K):
        for j in range(K):
            v = data[i, j]
            if v == 0:
                continue
            txt = f"{100 * v:.0f}" if normalize else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if v > thr else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    if normalize:
        cb.set_label("% de la fila (exhaustividad)")
    return save_fig(fig, out_dir, name)


def plot_per_class(metrics, classes, title, out_dir, name):
    pc = metrics["por_clase"]
    x = np.arange(len(classes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(classes)), 3.4))
    for k, (key, lab) in enumerate([("precision", "Precisión"), ("exhaustividad", "Exhaustividad"), ("f1", "F1")]):
        vals = [pc[c][key] for c in classes]
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=SERIES[k], label=lab)
    ax.axhline(1 / len(classes), color=INK2, linestyle="--", linewidth=1)
    ax.text(len(classes) - 0.5, 1 / len(classes) + 0.02, "azar (1/K)", color=INK2, fontsize=8, ha="right")
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, pad=26)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    return save_fig(fig, out_dir, name)


def plot_reliability(metrics, conf, correct, title, out_dir, name):
    tab = metrics["tabla_calibracion"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK2, linestyle="--", linewidth=1, label="Calibración perfecta")
    xs = [(lo + hi) / 2 for lo, hi, n, a, c in tab if n > 0]
    ys = [a for lo, hi, n, a, c in tab if n > 0]
    ax.bar(xs, ys, width=0.09, color=SERIES[0], label="Exactitud observada")
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Exactitud")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Diagrama de confiabilidad (ECE = {metrics['ece']:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    ax.hist(conf[correct == 1], bins=bins, color=SERIES[2], alpha=0.85, label="Aciertos", rwidth=0.9)
    ax.hist(conf[correct == 0], bins=bins, color=SERIES[1], alpha=0.75, label="Errores", rwidth=0.9)
    ax.set_xlabel("Confianza (probabilidad máxima)")
    ax.set_ylabel("Videos")
    ax.set_title("Confianza en aciertos y errores")
    ax.legend()
    fig.suptitle(title, y=1.03, fontsize=10, color=INK2)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def plot_roc_pr(y, proba, classes, title, out_dir, name):
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    K = len(classes)
    if len(np.unique(y)) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    onehot = np.eye(K)[y]
    # micro-promedio (una sola curva legible) + clases en gris tenue
    for i in range(K):
        if onehot[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(onehot[:, i], proba[:, i])
        axes[0].plot(fpr, tpr, color=GRID, linewidth=1)
        pr, rc, _ = precision_recall_curve(onehot[:, i], proba[:, i])
        axes[1].plot(rc, pr, color=GRID, linewidth=1)
    fpr, tpr, _ = roc_curve(onehot.ravel(), proba.ravel())
    axes[0].plot(fpr, tpr, color=SERIES[0], label=f"micro-promedio (AUC = {auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], color=INK2, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[0].set_title("Curva ROC (uno contra resto)")
    axes[0].legend(loc="lower right", fontsize=8)
    pr, rc, _ = precision_recall_curve(onehot.ravel(), proba.ravel())
    axes[1].plot(rc, pr, color=SERIES[1], label=f"micro-promedio (AUC = {auc(rc, pr):.3f})")
    axes[1].axhline(1 / K, color=INK2, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Exhaustividad")
    axes[1].set_ylabel("Precisión")
    axes[1].set_title("Curva precisión–exhaustividad")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.suptitle(title + "  (en gris: cada clase)", y=1.03, fontsize=10, color=INK2)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def write_metric_tables(m, classes, split, out_dir, coma):
    pct = lambda v: fmt_pct(v, coma=coma)
    num = lambda v: fmt_num(v, coma=coma)
    ci_a = m.get("exactitud_IC95")
    ci_f = m.get("f1_macro_IC95")
    rows = [
        ["Videos evaluados", m["n"]],
        ["Exactitud", pct(m["exactitud"]) + (f" [{pct(ci_a[0])}; {pct(ci_a[1])}]" if ci_a else "")],
        ["Exactitud balanceada", pct(m["exactitud_balanceada"])],
    ]
    for k in (2, 3, 5):
        if f"top{k}" in m:
            rows.append([f"Exactitud top-{k}", pct(m[f"top{k}"])])
    rows += [
        ["Precisión (macro)", pct(m["precision_macro"])],
        ["Exhaustividad (macro)", pct(m["exhaustividad_macro"])],
        ["$F_1$ (macro)", pct(m["f1_macro"]) + (f" [{pct(ci_f[0])}; {pct(ci_f[1])}]" if ci_f else "")],
        ["$F_1$ (ponderado)", pct(m["f1_ponderado"])],
        [r"Kappa de Cohen ($\kappa$)", num(m["kappa_cohen"])],
        ["Coef. de correlación de Matthews", num(m["mcc"])],
        ["Pérdida logarítmica", num(m["log_loss"])],
        ["Puntaje de Brier", num(m["brier"])],
    ]
    if "roc_auc_macro_ovr" in m:
        rows.append(["ROC-AUC (macro, uno contra resto)", num(m["roc_auc_macro_ovr"])])
    rows += [
        ["Error de calibración esperado (ECE)", num(m["ece"])],
        "MIDRULE",
        ["Línea base: azar ($1/K$)", pct(m["azar_1_sobre_K"])],
        ["Línea base: clase mayoritaria", pct(m["clase_mayoritaria"])],
        ["$p$-valor binomial (exactitud $>$ azar)", fmt_num(m["p_valor_binomial_vs_azar"], 4, coma)],
    ]
    nota = ("Entre corchetes, intervalo de confianza del 95\\,\\% por remuestreo bootstrap "
            f"({m.get('remuestreos', 0)} remuestreos). Con {m['n']} videos, cada video equivale a "
            f"{fmt_pct(1 / m['n'], coma=coma)} de exactitud.")
    latex_table(os.path.join(out_dir, "tablas", f"tabla_metricas_globales_{split}.tex"),
                f"Métricas globales del modelo sobre el conjunto de {split}.",
                f"tab:metricas-globales-{split}", ["Métrica", "Valor"], rows, align="lr", nota=nota)
    pc_rows = []
    for c in classes:
        d = m["por_clase"][c]
        pc_rows.append([tex_escape(c), pct(d["precision"]), pct(d["exhaustividad"]), pct(d["f1"]), d["soporte"]])
    pc_rows += ["MIDRULE",
                [r"\textit{Promedio macro}", pct(m["precision_macro"]), pct(m["exhaustividad_macro"]),
                 pct(m["f1_macro"]), m["n"]],
                [r"\textit{Promedio ponderado}", pct(m["precision_ponderado"]),
                 pct(m["exhaustividad_ponderado"]), pct(m["f1_ponderado"]), m["n"]]]
    latex_table(os.path.join(out_dir, "tablas", f"tabla_metricas_por_clase_{split}.tex"),
                f"Precisión, exhaustividad y $F_1$ por seña (conjunto de {split}).",
                f"tab:metricas-clase-{split}", ["Seña", "Precisión", "Exhaustividad", "$F_1$", "Soporte"],
                pc_rows, align="lrrrr")
    # matriz de confusión en LaTeX
    K = len(classes)
    abbrev = [c[:3] + "." for c in classes]
    cm_rows = [[tex_escape(classes[i])] + [str(v) if v else r"$\cdot$" for v in m["matriz_confusion"][i]]
               for i in range(K)]
    latex_table(os.path.join(out_dir, "tablas", f"tabla_matriz_confusion_{split}.tex"),
                f"Matriz de confusión (conjunto de {split}). Filas: clase real; columnas: clase predicha.",
                f"tab:matriz-confusion-{split}", ["Real / Pred."] + [tex_escape(a) for a in abbrev],
                cm_rows, align="l" + "c" * K)
    write_csv(os.path.join(out_dir, f"metricas_por_clase_{split}.csv"),
              ["clase", "precision", "exhaustividad", "f1", "soporte"],
              [[c, m["por_clase"][c]["precision"], m["por_clase"][c]["exhaustividad"], m["por_clase"][c]["f1"],
                m["por_clase"][c]["soporte"]] for c in classes])


def evaluate_split(args, model, rows, classes, split):
    sub = [r for r in rows if r["split"] == split]
    X, y, used = load_sequences(args.data_dir, sub, classes, args.seq_length, args.data_type)
    if X is None:
        log(f"  ({split}: sin secuencias.)")
        return None
    log(f"\n--- Partición '{split}': {len(X)} videos ---")
    proba = predict_proba(model, X)
    if proba.ndim == 3:  # por si el modelo devuelve secuencias
        proba = proba[:, -1]
    m, pred, conf = compute_metrics(y, proba, classes, args.bootstrap, args.seed)
    correct = (pred == y).astype(float)
    out = args.out_dir
    m["figuras"] = {
        "confusion": plot_confusion(m["matriz_confusion"], classes, f"Matriz de confusión — {split}", out,
                                    f"matriz_confusion_{split}"),
        "confusion_norm": plot_confusion(m["matriz_confusion"], classes,
                                         f"Matriz de confusión normalizada — {split}", out,
                                         f"matriz_confusion_normalizada_{split}", normalize=True),
        "por_clase": plot_per_class(m, classes, f"Métricas por seña — {split}", out, f"metricas_por_clase_{split}"),
        "calibracion": plot_reliability(m, conf, correct, f"Conjunto de {split}", out, f"calibracion_{split}"),
        "roc_pr": plot_roc_pr(y, proba, classes, f"Conjunto de {split}", out, f"roc_pr_{split}"),
    }
    write_metric_tables(m, classes, split, out, not args.decimal_punto)
    order = np.argsort(-proba, 1)
    pred_rows = []
    for i, r in enumerate(used):
        top3 = "; ".join(f"{classes[j]}={proba[i, j]:.3f}" for j in order[i, :3])
        pred_rows.append([r["stem"], classes[y[i]], classes[pred[i]], int(pred[i] == y[i]),
                          round(float(conf[i]), 4), top3, int(is_augmented_row(r["split"], r["stem"]))])
    write_csv(os.path.join(out, f"predicciones_{split}.csv"),
              ["video", "clase_real", "clase_predicha", "acierto", "confianza", "top3", "aumentado"], pred_rows)
    write_csv(os.path.join(out, f"probabilidades_{split}.csv"), ["video", "clase_real"] + [f"p_{c}" for c in classes],
              [[r["stem"], classes[y[i]]] + [round(float(v), 6) for v in proba[i]] for i, r in enumerate(used)])
    ci = m.get("exactitud_IC95", [float("nan")] * 2)
    log(f"  Exactitud: {100 * m['exactitud']:.1f} % ({m['aciertos']}/{m['n']})  IC95 [{100 * ci[0]:.1f}; "
        f"{100 * ci[1]:.1f}]  |  azar: {100 * m['azar_1_sobre_K']:.1f} %  |  p = {m['p_valor_binomial_vs_azar']:.4g}")
    log(f"  F1 macro: {m['f1_macro']:.3f}  kappa: {m['kappa_cohen']:.3f}  MCC: {m['mcc']:.3f}  "
        f"ECE: {m['ece']:.3f}  log-loss: {m['log_loss']:.3f}")
    if m["pares_mas_confundidos"]:
        log("  Pares más confundidos: " + ", ".join(f"{a}→{b} ({n})" for a, b, n in m["pares_mas_confundidos"][:5]))
    m["_X"], m["_y"], m["_used"] = X, y, used  # para embeddings (se eliminan antes de guardar)
    return m


# ---------------------------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------------------------
def plot_embeddings(args, keras_model, evals, classes):
    log("\n=== 4. Proyección de la representación interna ===")
    import tensorflow as tf
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        feat_model = tf.keras.Model(inputs=keras_model.inputs, outputs=keras_model.layers[-2].output)
    except Exception as e:
        log(f"  (No se pudo construir el submodelo: {e})")
        return None
    Xs, ys, ss = [], [], []
    for split, m in evals.items():
        Xs.append(m["_X"])
        ys.append(m["_y"])
        ss += [split] * len(m["_y"])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    ss = np.array(ss)
    Z = predict_proba(feat_model, X)
    Z = Z.reshape(len(Z), -1)
    p2 = PCA(n_components=2, random_state=args.seed).fit_transform(Z)
    perp = max(2, min(30, (len(Z) - 1) // 3))
    t2 = TSNE(n_components=2, perplexity=perp, random_state=args.seed, init="pca").fit_transform(Z)
    K = len(classes)
    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    markers = {"train": "o", "test": "^"}
    for ax, P, title in [(axes[0], p2, "PCA"), (axes[1], t2, f"t-SNE (perplejidad {perp})")]:
        for k in range(K):
            for s in np.unique(ss):
                msk = (y == k) & (ss == s)
                if msk.any():
                    ax.scatter(P[msk, 0], P[msk, 1], s=22 if s == "test" else 12, marker=markers.get(s, "s"),
                               color=cmap(k), edgecolor="white", linewidth=0.5,
                               label=classes[k] if s == np.unique(ss)[0] else None, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, markerscale=1.5)
    fig.suptitle("Representación de la penúltima capa (círculos: train, triángulos: test)",
                 fontsize=10, color=INK2)
    fig.tight_layout()
    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(Z, y)) if len(np.unique(y)) > 1 else None
    except Exception:
        sil = None
    log(f"  Coeficiente de silueta (penúltima capa, por clase): {sil}")
    return {"figura": save_fig(fig, args.out_dir, "embeddings_pca_tsne"), "silueta": sil,
            "capa": keras_model.layers[-2].name}


# ---------------------------------------------------------------------------
# 5. Validación cruzada
# ---------------------------------------------------------------------------
def build_model(input_shape, K, arch, lr):
    import tensorflow as tf
    L = tf.keras.layers
    if arch == "original":  # misma arquitectura que train_lstm_harp.py
        model = tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.LSTM(2048, return_sequences=True, dropout=0.5),
            L.Dense(512, activation="relu"),
            L.Dropout(0.5),
            L.LSTM(256, return_sequences=True),
            L.Dropout(0.5),
            L.LSTM(128, return_sequences=False),
            L.Dropout(0.5),
            L.Dense(K, activation="softmax"),
        ])
    else:  # "ligera": recomendada para pocos datos
        model = tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.LSTM(128, return_sequences=True, dropout=0.3),
            L.LSTM(64, dropout=0.3),
            L.Dense(64, activation="relu"),
            L.Dropout(0.5),
            L.Dense(K, activation="softmax"),
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def cross_validation(args, rows, classes, signers):
    log(f"\n=== 5. Validación cruzada ({args.cv} pliegues, arquitectura '{args.cv_arch}') ===")
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    tf.keras.utils.set_random_seed(args.seed)
    pool = [r for r in rows if r["split"] == "train" or (args.cv_incluir_test and r["split"] == "test")]
    X, y, used = load_sequences(args.data_dir, pool, classes, args.seq_length, args.data_type)
    if X is None:
        log("  (Sin secuencias.)")
        return None
    if args.cv_por_persona and signers:
        groups = np.array([signer_of(signers, r) for r in used])
        gname = "persona"
    else:
        groups = np.array([r["clase"] + "/" + video_original(r["split"], r["stem"]) for r in used])
        gname = "video original"
    n_groups = len(np.unique(groups))
    k = min(args.cv, n_groups)
    log(f"  {len(X)} secuencias, {n_groups} grupos por {gname}. Las copias aumentadas de un video "
        "quedan siempre en el mismo pliegue (sin fuga).")
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
    K = len(classes)
    folds, cm_total, oof_rows = [], np.zeros((K, K), dtype=int), []
    for f, (tr, va) in enumerate(skf.split(X, y, groups)):
        # Evaluar solo sobre originales del pliegue de validación (las copias no son datos nuevos)
        va_orig = np.array([i for i in va if not is_augmented_row(used[i]["split"], used[i]["stem"])]) \
            if args.cv_solo_originales else va
        if len(va_orig) == 0:
            va_orig = va
        tf.keras.backend.clear_session()
        model = build_model(X.shape[1:], K, args.cv_arch, args.cv_lr)
        model.fit(X[tr], np.eye(K)[y[tr]], epochs=args.cv_epochs, batch_size=32, verbose=0, shuffle=True)
        proba_va = predict_proba(model, X[va_orig])
        pr = proba_va.argmax(1)
        for i, p in zip(va_orig, proba_va):
            oof_rows.append([f + 1, used[i]["split"], used[i]["stem"], groups[i], classes[y[i]]]
                            + [round(float(v), 6) for v in p])
        acc = float((pr == y[va_orig]).mean())
        f1 = float(f1_score(y[va_orig], pr, labels=list(range(K)), average="macro", zero_division=0))
        cm_total += confusion_matrix(y[va_orig], pr, labels=list(range(K)))
        folds.append({"pliegue": f + 1, "n_train": int(len(tr)), "n_val": int(len(va_orig)),
                      "exactitud": acc, "f1_macro": f1})
        log(f"  Pliegue {f + 1}/{k}: exactitud {100 * acc:.1f} %  F1 macro {f1:.3f}  (n_val={len(va_orig)})")
    write_csv(os.path.join(args.out_dir, "probabilidades_validacion_cruzada.csv"),
              ["pliegue", "particion", "video", "grupo", "clase_real"] + [f"p_{c}" for c in classes], oof_rows)
    accs = np.array([d["exactitud"] for d in folds])
    f1s = np.array([d["f1_macro"] for d in folds])
    res = {"k": k, "agrupado_por": gname, "arquitectura": args.cv_arch, "epocas": args.cv_epochs,
           "pliegues": folds, "exactitud_media": float(accs.mean()), "exactitud_desvio": float(accs.std(ddof=1)) if k > 1 else 0.0,
           "f1_macro_media": float(f1s.mean()), "f1_macro_desvio": float(f1s.std(ddof=1)) if k > 1 else 0.0,
           "matriz_confusion_agregada": cm_total.tolist()}
    log(f"  Exactitud media: {100 * accs.mean():.1f} % ± {100 * res['exactitud_desvio']:.1f}   "
        f"F1 macro: {f1s.mean():.3f} ± {res['f1_macro_desvio']:.3f}")
    res["figura_confusion"] = plot_confusion(cm_total, classes, f"Validación cruzada ({k} pliegues) — agregada",
                                             args.out_dir, "matriz_confusion_validacion_cruzada")
    coma = not args.decimal_punto
    rows_t = [[d["pliegue"], d["n_train"], d["n_val"], fmt_pct(d["exactitud"], coma=coma),
               fmt_pct(d["f1_macro"], coma=coma)] for d in folds]
    rows_t += ["MIDRULE", [r"\textbf{Media $\pm$ desvío}", "", "",
                           f"{fmt_pct(accs.mean(), coma=coma)} $\\pm$ {fmt_pct(res['exactitud_desvio'], coma=coma)}",
                           f"{fmt_pct(f1s.mean(), coma=coma)} $\\pm$ {fmt_pct(res['f1_macro_desvio'], coma=coma)}"]]
    latex_table(os.path.join(args.out_dir, "tablas", "tabla_validacion_cruzada.tex"),
                f"Validación cruzada estratificada de {k} pliegues agrupada por {gname}.",
                "tab:validacion-cruzada", ["Pliegue", "Entren.", "Valid.", "Exactitud", "$F_1$ macro"],
                rows_t, align="rrrrr")
    return res


# ---------------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------------
def write_markdown(path, summary):
    L = ["# Informe de métricas — LSPy (Inception V3 + LSTM)", "",
         f"Generado: {summary['fecha']}  ", f"Comando: `{summary['comando']}`", ""]
    a = summary.get("auditoria", {})
    if a:
        L += ["## 1. Conjunto de datos", ""]
        t = a.get("conteos", {}).get("totales", {})
        L.append(f"- Clases: {', '.join(a.get('conteos', {}).get('clases', []))}")
        L.append("- Totales: " + ", ".join(f"{k}={v}" for k, v in sorted(t.items())))
        g = a.get("grupos", {})
        if g:
            L.append(f"- Grupos ({g['tipo']}) compartidos entre train y test: {g['compartidos_train_test'] or 'ninguno'}")
        L.append(f"- Videos originales repetidos en train y test: "
                 f"{a.get('fuga_mismo_video_original_en_train_y_test') or 'ninguno'}")
        if "colisiones_glob" in a:
            L.append(f"- Secuencias contaminadas por el glob: {a['colisiones_glob']['cantidad']}")
        if "cobertura_global" in a:
            c = a["cobertura_global"]
            L.append(f"- Fotogramas con mano por video: media {c['media']:.1f}, mediana {c['mediana']:.0f}, "
                     f"mín {c['min']}, máx {c['max']}")
        if "caracteristicas" in a:
            c = a["caracteristicas"]
            L.append(f"- Características: forma {c['forma']}, ¿softmax? {c['parecen_probabilidades_softmax']}, "
                     f"pasos idénticos {100 * c['proporcion_pasos_identicos']:.1f} %")
        if a.get("fig_cobertura"):
            L += ["", f"![cobertura]({a['fig_cobertura'].replace('.pdf', '.png')})"]
        L.append("")
    tr = summary.get("entrenamiento", {})
    if tr:
        L += ["## 2. Entrenamiento", "", "```", json.dumps(tr, indent=2, ensure_ascii=False), "```"]
        if tr.get("figura"):
            L.append(f"![curvas]({tr['figura'].replace('.pdf', '.png')})")
        L.append("")
    if summary.get("modelo"):
        L += ["## 3. Modelo", "", "```", json.dumps(summary["modelo"], indent=2, ensure_ascii=False), "```", ""]
    for split, m in summary.get("evaluacion", {}).items():
        L += [f"### Evaluación — {split}", "",
              "| Métrica | Valor |", "|---|---|"]
        for key, lab in [("n", "Videos"), ("exactitud", "Exactitud"), ("exactitud_balanceada", "Exactitud balanceada"),
                         ("top2", "Top-2"), ("top3", "Top-3"), ("precision_macro", "Precisión macro"),
                         ("exhaustividad_macro", "Exhaustividad macro"), ("f1_macro", "F1 macro"),
                         ("kappa_cohen", "Kappa de Cohen"), ("mcc", "MCC"), ("log_loss", "Log-loss"),
                         ("brier", "Brier"), ("roc_auc_macro_ovr", "ROC-AUC macro"), ("ece", "ECE"),
                         ("azar_1_sobre_K", "Azar 1/K"), ("p_valor_binomial_vs_azar", "p-valor vs azar")]:
            if key in m:
                v = m[key]
                L.append(f"| {lab} | {v:.4f} |" if isinstance(v, float) else f"| {lab} | {v} |")
        if "exactitud_IC95" in m:
            L.append(f"| IC95 exactitud | [{m['exactitud_IC95'][0]:.3f}; {m['exactitud_IC95'][1]:.3f}] |")
            L.append(f"| IC95 F1 macro | [{m['f1_macro_IC95'][0]:.3f}; {m['f1_macro_IC95'][1]:.3f}] |")
        L.append("")
        for k_, f_ in m.get("figuras", {}).items():
            if f_:
                L.append(f"![{k_}]({f_.replace('.pdf', '.png')})")
        L.append("")
    if summary.get("embeddings"):
        e = summary["embeddings"]
        L += ["## 4. Embeddings", "", f"Silueta: {e.get('silueta')}", "",
              f"![emb]({e['figura'].replace('.pdf', '.png')})", ""]
    if summary.get("validacion_cruzada"):
        c = summary["validacion_cruzada"]
        L += ["## 5. Validación cruzada", "",
              f"{c['k']} pliegues agrupados por {c['agrupado_por']}: exactitud {100 * c['exactitud_media']:.1f} % "
              f"± {100 * c['exactitud_desvio']:.1f}; F1 macro {c['f1_macro_media']:.3f} ± {c['f1_macro_desvio']:.3f}", ""]
    L += ["## Archivos para la tesis", "", "- Tablas LaTeX: `tablas/*.tex` (usar `\\input{...}`, requiere booktabs).",
          "- Figuras: `figuras/*.pdf` (vectoriales) y `.png`.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Métricas completas para la tesis LSPy.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="data", help="Directorio de salida de handtrack.py")
    p.add_argument("--model", default="lstm_senha_model",
                   help="Modelo (SavedModel, .h5/.hdf5/.keras) o 'mejor' para el checkpoint con menor val_loss")
    p.add_argument("--sin-modelo", action="store_true", help="Solo auditoría de datos y curvas")
    p.add_argument("--splits", nargs="+", default=["test", "train"], help="Particiones a evaluar")
    p.add_argument("--seq-length", type=int, default=150)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--class-limit", type=int, default=10)
    p.add_argument("--clases", nargs="+", default=None,
                   help="Subconjunto de señas (por defecto, las del .json del modelo o todas)")
    p.add_argument("--data-type", default=None,
                   help="features2048 | probs | landmarks | features (versión anterior). "
                        "Por defecto, la del .json del modelo o features2048")
    p.add_argument("--logs-dir", default=os.path.join("data", "logs"))
    p.add_argument("--log-file", default=None, help="Log de CSVLogger específico (por defecto el más reciente)")
    p.add_argument("--raw-dir", default=None, help="Carpeta de videos crudos (p. ej. rawdata_clean) para fps/duración")
    p.add_argument("--signers-csv", default=None,
                   help="CSV 'video_original,persona' para análisis y validación cruzada por persona")
    p.add_argument("--out-dir", default=None, help="Carpeta de salida (por defecto metricas/<fecha_hora>)")
    p.add_argument("--bootstrap", type=int, default=2000, help="Remuestreos para IC95 (0 = desactivar)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embeddings", action="store_true", help="Proyección PCA/t-SNE de la penúltima capa")
    p.add_argument("--latencia-inception", action="store_true",
                   help="Medir el tiempo de InceptionV3 por fotograma (descarga pesos de ImageNet)")
    p.add_argument("--decimal-punto", action="store_true", help="Usar punto decimal en LaTeX (por defecto coma)")
    p.add_argument("--cv", type=int, default=0, help="Pliegues de validación cruzada (0 = no ejecutar)")
    p.add_argument("--cv-epochs", type=int, default=60)
    p.add_argument("--cv-lr", type=float, default=1e-4)
    p.add_argument("--cv-arch", choices=["original", "ligera"], default="original")
    p.add_argument("--cv-por-persona", action="store_true", help="Agrupar pliegues por persona (requiere --signers-csv)")
    p.add_argument("--cv-incluir-test", action="store_true", help="Incluir la partición test en el conjunto de CV")
    p.add_argument("--cv-solo-originales", action="store_true",
                   help="Evaluar cada pliegue solo con videos originales (no con copias aumentadas)")
    return p.parse_args()


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("metricas", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(args.out_dir, "tablas"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figuras"), exist_ok=True)
    log(f"Salida en: {os.path.abspath(args.out_dir)}")

    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] Falta scikit-learn:  pip install scikit-learn==1.5.2")

    if args.data_type is None:
        meta = None if args.sin_modelo or args.model == "mejor" else load_meta(args.model)
        args.data_type = meta["data_type"] if meta else "features2048"
        log(f"Representación de entrada: {args.data_type}")

    all_rows = read_data_file(args.data_dir)
    if args.clases is None and not args.sin_modelo and args.model != "mejor":
        meta = load_meta(args.model)
        args.clases = meta.get("classes") if meta else None
    rows, classes = filter_rows(all_rows, args.seq_length, args.max_frames, args.class_limit, args.clases)
    if len(rows) < len(all_rows):
        log(f"[aviso] {len(all_rows) - len(rows)} filas de data_file.csv quedan fuera por el filtro "
            f"(seq_length/max_frames/class_limit), igual que en DataSet.")
    signers = load_signers_csv(args.signers_csv)
    summary = {"fecha": datetime.now().isoformat(timespec="seconds"), "comando": " ".join(sys.argv),
               "parametros": {k: v for k, v in vars(args).items()}}

    summary["auditoria"] = audit_dataset(args, rows, classes, signers)
    summary["entrenamiento"] = training_curves(args)

    if not args.sin_modelo:
        log("\n=== 3. Evaluación del modelo ===")
        mpath = resolve_model_path(args.model, args.data_dir)
        if not os.path.exists(mpath):
            sys.exit(f"[ERROR] No existe el modelo '{mpath}'. Use --model o --sin-modelo.")
        model, keras_model = load_model_any(mpath)
        summary["modelo"] = {"ruta": mpath, **model_info(keras_model, mpath)}
        evals = {}
        for s in args.splits:
            m = evaluate_split(args, model, rows, classes, s)
            if m is not None:
                evals[s] = m
        if evals:
            first = next(iter(evals.values()))
            summary["modelo"]["latencia_lstm_por_secuencia"] = measure_latency(model, first["_X"][0])
            log(f"  Latencia LSTM por video: {summary['modelo']['latencia_lstm_por_secuencia']['media_ms']:.1f} ms")
        if args.latencia_inception:
            li = inception_latency()
            li["estimado_150_fotogramas_lote30_s"] = li["por_fotograma_lote30_ms"] * args.seq_length / 1000
            summary["modelo"]["latencia_inception"] = li
            log(f"  InceptionV3: {li['por_fotograma_lote1_ms']:.1f} ms/fotograma (lote 1), "
                f"{li['por_fotograma_lote30_ms']:.1f} ms (lote 30) → ~{li['estimado_150_fotogramas_lote30_s']:.1f} s "
                f"por video de {args.seq_length} fotogramas")
        if args.embeddings and keras_model is not None and evals:
            summary["embeddings"] = plot_embeddings(args, keras_model, evals, classes)
        # tabla comparativa de particiones
        if evals:
            coma = not args.decimal_punto
            cmp_rows = [[s, m["n"], fmt_pct(m["exactitud"], coma=coma), fmt_pct(m["f1_macro"], coma=coma),
                         fmt_num(m["kappa_cohen"], coma=coma), fmt_pct(m["azar_1_sobre_K"], coma=coma)]
                        for s, m in evals.items()]
            latex_table(os.path.join(args.out_dir, "tablas", "tabla_resumen_particiones.tex"),
                        "Resumen del desempeño por partición.", "tab:resumen-particiones",
                        ["Partición", "Videos", "Exactitud", "$F_1$ macro", r"$\kappa$", "Azar"], cmp_rows,
                        align="lrrrrr",
                        nota="Una diferencia grande entre entrenamiento y prueba indica sobreajuste.")
        summary["evaluacion"] = evals

    if args.cv:
        summary["validacion_cruzada"] = cross_validation(args, rows, classes, signers)

    summary = strip_private(summary)
    with open(os.path.join(args.out_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    write_markdown(os.path.join(args.out_dir, "resumen.md"), summary)
    log(f"\nListo. Revise {os.path.join(args.out_dir, 'resumen.md')} y las carpetas tablas/ y figuras/.")


if __name__ == "__main__":
    main()
