"""
Demo del prototipo: carga el agente DRL entrenado y la heuristica optimizada de
un escenario, simula una temporada de validacion con ambos y grafica el
calendario de riego y la humedad del suelo.

Uso:
  python scripts/demo.py --escenario tope_75 --anio 85
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from riego.clima import cargar_clima, separar_por_anio  # noqa: E402
from riego.entorno import EntornoRiego  # noqa: E402
from riego.escenarios import ESCENARIOS  # noqa: E402


def correr(env, politica):
    obs, _ = env.reset()
    dias, humedad, lluvia, terminado = [], [], [], False
    while not terminado:
        obs, _, terminado, _, info = env.step(politica(obs))
        c = env.model._init_cond
        dias.append(c.dap)
        humedad.append(100 * (1 - c.depletion / c.taw) if c.taw > 0 else np.nan)
    # riego realmente aplicado por AquaCrop (respeta el tope estacional)
    w = env.model.get_water_flux()
    w = w[w.dap > 0]
    riego = np.zeros(int(w.dap.max()) + 1)
    riego[w.dap.astype(int).values] = w.IrrDay.values
    return info, riego, dias, humedad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--escenario", default="tope_75", choices=list(ESCENARIOS))
    p.add_argument("--anio", type=int, default=85, help="ano de validacion (71-100)")
    a = p.parse_args()

    res = RAIZ / "results" / a.escenario
    clima = separar_por_anio(cargar_clima())
    base = ESCENARIOS[a.escenario]

    # Agente DRL
    modelo = PPO.load(res / "agente_ppo", device="cpu")
    norm = pickle.load(open(res / "normalizacion_obs.pkl", "rb"))
    env_drl = EntornoRiego(clima, {**base, "accion": "depth_discreet", "dias_decision": 3})
    env_drl.anio_fijo = a.anio
    drl = correr(env_drl, lambda o: int(modelo.predict(norm.normalize_obs(o[None]), deterministic=True)[0][0]))

    # Heuristica optimizada
    umbrales = np.array(json.load(open(res / "heuristica.json"))["umbrales"])
    env_h = EntornoRiego(clima, {**base, "accion": "smt4", "dias_decision": 1})
    env_h.anio_fijo = a.anio
    heu = correr(env_h, lambda o: umbrales)

    for nombre, (info, riego, *_) in [("DRL (PPO)", drl), ("Heuristica", heu)]:
        print(f"{nombre:<11} rendimiento={info['rendimiento']:.2f} t/ha  riego={info['riego']:.0f} mm  "
              f"ganancia=${info['ganancia']:.1f}/ha")

    lluvia = clima[a.anio]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for nombre, (info, riego, *_), col in [("DRL", drl, "#c4511f"), ("Heuristica", heu, "#1f5f8b")]:
        d = np.nonzero(riego)[0]
        ax[0].bar(d + (0.3 if nombre == "DRL" else -0.3), riego[d], width=0.6, color=col,
                  label=f"{nombre}: {info['riego']:.0f} mm, ${info['ganancia']:.0f}/ha")
    pp = lluvia.Precipitation.values[120:120 + len(drl[1])] if base["lluvia"] else None
    if pp is not None:
        ax[0].bar(np.arange(len(pp)), -pp, color="#7fb3d5", width=0.8, label="Lluvia (negativo)")
    ax[0].set_ylabel("mm/dia")
    ax[0].set_title(f"Escenario {a.escenario}, ano de validacion {a.anio}")
    ax[0].legend(fontsize=8)
    ax[1].plot(drl[2], drl[3], color="#c4511f", label="DRL")
    ax[1].plot(heu[2], heu[3], color="#1f5f8b", label="Heuristica")
    ax[1].set_ylabel("Humedad zona radicular (% TAW)")
    ax[1].set_xlabel("Dias despues de la siembra")
    ax[1].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    out = RAIZ / "figures" / f"demo_{a.escenario}_anio{a.anio}.png"
    fig.savefig(out, dpi=150)
    print("figura:", out)


if __name__ == "__main__":
    main()
