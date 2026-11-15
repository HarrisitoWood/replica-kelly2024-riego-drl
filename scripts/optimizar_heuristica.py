"""
Optimiza la heuristica de referencia: 4 umbrales de humedad del suelo (uno por
etapa de crecimiento) que maximizan la ganancia media en los 70 anos de
entrenamiento. Igual que el paper (seccion 2.5) y el notebook
optimize_smt_benchmarks.ipynb del repositorio original: evolucion diferencial + refinamiento local Nelder-Mead.

Uso:
  python scripts/optimizar_heuristica.py --escenario con_lluvia
  python scripts/optimizar_heuristica.py --escenario tope_75
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from riego.clima import cargar_clima, separar_por_anio  # noqa: E402
from riego.escenarios import ESCENARIOS  # noqa: E402
from riego.evaluacion import ANIOS_TRAIN, evaluar, politica_umbrales, simular_temporadas  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--escenario", required=True, choices=list(ESCENARIOS))
    p.add_argument("--popsize", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maxfev-local", type=int, default=40)
    a = p.parse_args()

    clima = separar_por_anio(cargar_clima())
    cfg = {**ESCENARIOS[a.escenario], "accion": "smt4"}

    historial = []
    out = RAIZ / "results" / a.escenario
    out.mkdir(parents=True, exist_ok=True)
    # cache en disco: si el proceso se corta, al relanzarlo las evaluaciones ya
    # hechas se recuperan (la evolucion diferencial con semilla fija es determinista)
    f_cache = out / "cache_heuristica.json"
    cache = json.load(open(f_cache)) if f_cache.exists() else {}

    def objetivo(x):
        k = ",".join(f"{v:.10f}" for v in x)
        if k in cache:
            g = cache[k]
        else:
            g = float(simular_temporadas(politica_umbrales(x), clima, cfg, ANIOS_TRAIN).ganancia.mean())
            cache[k] = g
            json.dump(cache, open(f_cache, "w"))
        historial.append([*map(float, x), float(g)])
        print(f"  umbrales={np.round(x, 3)} ganancia_train={g:.1f}", flush=True)
        return -g

    t0 = time.time()
    res = differential_evolution(objetivo, bounds=[(0, 1)] * 4, popsize=a.popsize,
                                 maxiter=a.maxiter, tol=0.1, seed=a.seed, polish=False)
    # refinamiento local desde el mejor punto de la evolucion diferencial
    x0 = np.clip(res.x, 0, 1)
    simplex = np.vstack([x0] + [np.clip(x0 + 0.1 * np.eye(4)[i] * (1 if x0[i] < 0.9 else -1), 0, 1)
                                for i in range(4)])
    loc = minimize(lambda x: objetivo(np.clip(x, 0, 1)), x0, method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, maxfev=a.maxfev_local, xatol=0.01, fatol=0.5))
    mejor = np.clip(loc.x, 0, 1) if loc.fun < res.fun else x0
    tr, va, detalle = evaluar(politica_umbrales(mejor), clima, cfg)

    detalle.to_csv(out / "heuristica_por_anio.csv", index=False)
    json.dump(dict(escenario=a.escenario, umbrales=list(map(float, mejor)),
                   ganancia_train=tr, ganancia_val=va, evaluaciones=len(historial),
                   minutos=(time.time() - t0) / 60, historial=historial),
              open(out / "heuristica.json", "w"), indent=2)
    print(f"[{a.escenario}] umbrales={np.round(mejor, 3)} train=${tr:.1f}/ha val=${va:.1f}/ha")


if __name__ == "__main__":
    main()
