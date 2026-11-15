"""
Junta los resultados de todos los escenarios, arma las tablas comparables con
las Tablas 4 y 5 del paper y genera las figuras.

Uso:  python scripts/resumen.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from riego.escenarios import ESCENARIOS, PAPER  # noqa: E402

RES, FIG = RAIZ / "results", RAIZ / "figures"
FIG.mkdir(exist_ok=True)
NOMBRES = {"con_lluvia": "Con lluvia, sin tope", "sin_lluvia": "Sin lluvia, sin tope",
           "tope_125": "Tope 125 mm", "tope_100": "Tope 100 mm", "tope_75": "Tope 75 mm"}


def leer(esc, archivo):
    f = RES / esc / archivo
    return json.load(open(f)) if f.exists() else None


filas = []
for esc in ESCENARIOS:
    h, d = leer(esc, "heuristica.json"), leer(esc, "drl.json")
    if not (h and d):
        continue
    mejora = d["drl_val"] - h["ganancia_val"]
    filas.append({
        "escenario": NOMBRES[esc],
        "heuristica_train": round(h["ganancia_train"], 1),
        "heuristica_val": round(h["ganancia_val"], 1),
        "drl_train": round(d["drl_train"], 1),
        "drl_val": round(d["drl_val"], 1),
        "mejora_val_$": round(mejora, 1),
        "mejora_val_%": round(100 * mejora / abs(h["ganancia_val"]), 1),
        "paper_drl_val": PAPER[esc]["drl_val"],
        "paper_mejora_val_$": PAPER[esc]["mejora_val"],
        "umbrales_heuristica": [round(x, 2) for x in h["umbrales"]],
    })
tabla = pd.DataFrame(filas)
tabla.to_csv(RES / "resumen.csv", index=False)
with open(RES / "resumen.md", "w") as f:
    f.write(tabla.drop(columns="umbrales_heuristica").to_markdown(index=False))
print(tabla.drop(columns="umbrales_heuristica").to_string(index=False))

# Figura 1: curvas de aprendizaje (equivalente a la Fig. 3 del paper)
esc_ok = [e for e in ESCENARIOS if (RES / e / "curva_aprendizaje.csv").exists()]
fig, ejes = plt.subplots(1, len(esc_ok), figsize=(4.2 * len(esc_ok), 3.6), sharey=False, squeeze=False)
for ax, esc in zip(ejes[0], esc_ok):
    c = pd.read_csv(RES / esc / "curva_aprendizaje.csv")
    ax.plot(c.pasos / 1e3, c.ganancia_train, color="#9aa5b1", lw=1, label="DRL entrenamiento")
    ax.plot(c.pasos / 1e3, c.ganancia_val, color="#f2a65a", lw=1, label="DRL validacion")
    ax.plot(c.pasos / 1e3, c.val_media10, color="#c4511f", lw=2, label="Validacion (media 10)")
    h = leer(esc, "heuristica.json")
    if h:
        ax.axhline(h["ganancia_val"], color="#1f5f8b", ls="--", lw=1.5, label="Heuristica (validacion)")
    lo = min(c.ganancia_val.quantile(0.2), h["ganancia_val"] if h else 0) - 40
    ax.set_ylim(lo, max(c.ganancia_train.max(), h["ganancia_val"] if h else 0) + 20)
    ax.set_title(NOMBRES[esc], fontsize=10)
    ax.set_xlabel("Pasos de entrenamiento (miles)")
    ax.grid(alpha=0.3)
ejes[0][0].set_ylabel("Ganancia media ($/ha)")
ejes[0][0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "curvas_aprendizaje.png", dpi=150)

# Figura 2: valor agregado del DRL vs heuristica (replica vs paper)
if len(tabla):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = range(len(tabla))
    ax.bar([i - 0.2 for i in x], tabla["mejora_val_$"], width=0.4, color="#c4511f", label="Esta replica")
    ax.bar([i + 0.2 for i in x], tabla["paper_mejora_val_$"], width=0.4, color="#9aa5b1", label="Kelly et al. (2024)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x), tabla.escenario, fontsize=9)
    ax.set_ylabel("DRL - heuristica en validacion ($/ha)")
    ax.set_title("Valor agregado del DRL frente a la heuristica optimizada")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "valor_agregado_drl.png", dpi=150)
print("figuras guardadas en", FIG)
