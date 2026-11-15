"""
Escenarios experimentales del paper (secciones 3.2 y 3.3):
  con_lluvia  -> clima normal, sin tope de agua          (Tabla 4, fila 1 / Tabla 5 "No cap")
  sin_lluvia  -> lluvia = 0 en entrenamiento y validacion (Tabla 4, fila 2)
  tope_125/100/75 -> tope estacional de riego en mm/ha    (Tabla 5)
Todos usan la configuracion ganadora del paper: laminas discretas cada 3 dias.
"""
ESCENARIOS = {
    "con_lluvia": dict(lluvia=True, tope_estacional=10_000),
    "sin_lluvia": dict(lluvia=False, tope_estacional=10_000),
    "tope_125": dict(lluvia=True, tope_estacional=125),
    "tope_100": dict(lluvia=True, tope_estacional=100),
    "tope_75": dict(lluvia=True, tope_estacional=75),
}

# Resultados publicados (Kelly et al., 2024) para comparar
PAPER = {
    "con_lluvia": dict(drl_val=520.1, mejora_val=-1.27, heur_val=521.4),
    "sin_lluvia": dict(drl_val=307.0, mejora_val=6.38, heur_val=300.6),
    "tope_125": dict(drl_val=332.0, mejora_val=4.3),
    "tope_100": dict(drl_val=286.5, mejora_val=7.1),
    "tope_75": dict(drl_val=228.0, mejora_val=12.2),
}
