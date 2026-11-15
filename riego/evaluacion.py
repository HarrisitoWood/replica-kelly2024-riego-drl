"""Evaluacion de estrategias en los 70 anos de entrenamiento y 30 de validacion."""
import numpy as np
import pandas as pd

from .entorno import EntornoRiego

ANIOS_TRAIN = list(range(1, 71))
ANIOS_VAL = list(range(71, 101))


def simular_temporadas(politica, clima, config, anios):
    """Corre una temporada por ano con `politica(obs) -> accion`.
    Devuelve un DataFrame con anio, rendimiento, riego y ganancia."""
    env = EntornoRiego(clima, config)
    filas = []
    for a in anios:
        env.anio_fijo = a
        obs, _ = env.reset()
        terminado = False
        while not terminado:
            obs, _, terminado, _, info = env.step(politica(obs))
        filas.append(info)
    return pd.DataFrame(filas)


def evaluar(politica, clima, config):
    """Ganancia media ($/ha) en entrenamiento y validacion + detalle por ano."""
    tr = simular_temporadas(politica, clima, config, ANIOS_TRAIN).assign(conjunto="entrenamiento")
    va = simular_temporadas(politica, clima, config, ANIOS_VAL).assign(conjunto="validacion")
    return tr.ganancia.mean(), va.ganancia.mean(), pd.concat([tr, va], ignore_index=True)


def politica_umbrales(umbrales):
    """Heuristica de 4 umbrales de humedad (fraccion de TAW), uno por etapa."""
    u = np.asarray(umbrales, dtype=float)
    return lambda obs: u
