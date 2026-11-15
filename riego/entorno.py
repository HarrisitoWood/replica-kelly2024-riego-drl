"""
Entorno de riego estilo Gymnasium sobre AquaCrop-OSPy (v3).

Es una traduccion de `aquacropgym/envs.py` (Kelly et al., 2024) a la API
actual de AquaCrop-OSPy y Gymnasium. Un episodio = una temporada de maiz
(siembra 1 de mayo) en 1 ha de suelo franco-arcillo-limoso en Champion, Nebraska.

Cada `dias_decision` dias el agente recibe una observacion del estado del
cultivo y decide una lamina de riego que se aplica al dia siguiente.
Al final de la temporada recibe como recompensa la ganancia ($/ha) / 1000.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil
from aquacrop.entities.co2 import CO2

# Parametros economicos del paper (seccion 2.3)
PRECIO_CULTIVO = 180.0   # $/tonelada
COSTO_RIEGO = 1.0        # $/ha-mm
COSTO_FIJO = 1728.0      # $/ha

LAMINAS = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]  # accion "discreet depth" (Tabla 1)

CONFIG_BASE = dict(
    anios=list(range(1, 71)),   # anos sinteticos usados al muestrear episodios
    dias_decision=3,            # intervalo de decision (el mejor en la Tabla 3)
    lluvia=True,                # False = escenario "sin lluvia"
    tope_estacional=10_000,     # mm/ha por temporada (10 000 = sin tope)
    riego_max_evento=25,        # mm por evento
    accion="depth_discreet",    # "depth_discreet" (DRL) o "smt4" (heuristica)
    siembra="05/01",
    cultivo="Maize",
    suelo="SiltClayLoam",
    humedad_inicial=70,         # % de capacidad de campo al inicio
    co2=363.8,
    anio_calendario=1995,
)


def ganancia(rend_t_ha, riego_mm):
    """Ganancia estacional en $/ha (misma formula del paper)."""
    return PRECIO_CULTIVO * rend_t_ha - COSTO_RIEGO * riego_mm - COSTO_FIJO


class EntornoRiego(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, clima_por_anio, config=None, seed=None):
        super().__init__()
        self.cfg = {**CONFIG_BASE, **(config or {})}
        self.clima = clima_por_anio
        self.rng = np.random.default_rng(seed)
        self.anio_fijo = None  # para evaluacion: fuerza un ano concreto

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
        if self.cfg["accion"] == "depth_discreet":
            self.action_space = spaces.Discrete(len(LAMINAS))
        elif self.cfg["accion"] == "smt4":
            # 4 umbrales de humedad (fraccion de TAW), uno por etapa de crecimiento
            self.action_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        else:
            raise ValueError(self.cfg["accion"])

        self._crop = Crop(self.cfg["cultivo"], planting_date=self.cfg["siembra"])
        self._soil = Soil(self.cfg["suelo"])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        anio = self.anio_fijo if self.anio_fijo is not None else int(self.rng.choice(self.cfg["anios"]))
        self.anio = anio
        wdf = self.clima[anio].copy()
        if not self.cfg["lluvia"]:
            wdf["Precipitation"] = 0.0

        y = self.cfg["anio_calendario"]
        m, d = self.cfg["siembra"].split("/")
        self.model = AquaCropModel(
            f"{y}/{m}/{d}", f"{y}/12/31", wdf, self._soil, self._crop,
            initial_water_content=InitialWaterContent(wc_type="Pct", value=[self.cfg["humedad_inicial"]]),
            irrigation_management=IrrigationManagement(
                irrigation_method=5, MaxIrrSeason=float(self.cfg["tope_estacional"]),
                MaxIrr=self.cfg["riego_max_evento"]),
            co2_concentration=CO2(constant_conc=True, current_concentration=self.cfg["co2"]),
        )
        self.model._initialize()
        self._w = self.model._weather[:, 2:4].astype(float)  # precipitacion, ETo
        self.riegos = []
        self.model.run_model(num_steps=1, initialize_model=False)  # dayshift=1 del original
        return self._obs(), {}

    # ------------------------------------------------------------ observacion
    def _obs(self):
        c = self.model._init_cond
        t = self.model._clock_struct.time_step_counter
        dep = c.depletion / c.taw if c.taw > 0 else 0.0
        ult7 = self._w[max(0, t - 7):t].mean(axis=0) if t > 0 else np.zeros(2)
        temporada = self._w[max(0, t - c.dap):t].sum(axis=0) if t > 0 else np.zeros(2)
        ayer = self._w[max(0, t - 1):t].reshape(-1) if t > 0 else np.zeros(2)
        fecha = self.model._clock_struct.time_span[min(t, len(self.model._clock_struct.time_span) - 1)]
        gs = np.zeros(4)
        gs[int(np.clip(int(c.growth_stage) - 1, 0, 3))] = 1
        obs = np.concatenate([
            [fecha.day, fecha.month, dep, c.dap, c.irr_cum, c.canopy_cover, c.biomass,
             self.cfg["tope_estacional"] - c.irr_cum],
            ult7, temporada, ayer, gs,
        ]).astype(np.float32)
        return np.nan_to_num(obs)

    # ------------------------------------------------------------------- step
    def step(self, action):
        clk = self.model._clock_struct
        irr = self.model._param_struct.IrrMngt
        if self.cfg["accion"] == "depth_discreet":
            irr.depth = LAMINAS[int(action)]
        else:
            umbrales = np.clip(np.asarray(action, dtype=float), 0, 1)

        for _ in range(self.cfg["dias_decision"]):
            if self.cfg["accion"] == "smt4":
                c = self.model._init_cond
                dep = c.depletion / c.taw if c.taw > 0 else 0.0
                gs = int(c.growth_stage) - 1
                if 0 <= gs <= 3 and (1 - dep) < umbrales[gs]:
                    irr.depth = float(np.clip(c.depletion, 0, self.cfg["riego_max_evento"]))
                else:
                    irr.depth = 0.0
            self.riegos.append(irr.depth)
            self.model.run_model(num_steps=1, initialize_model=False)
            if self.cfg["accion"] == "depth_discreet":
                irr.depth = 0.0  # la lamina elegida solo se aplica un dia
            if clk.model_is_finished or self._temporada_terminada():
                break

        terminado = clk.model_is_finished or self._temporada_terminada()
        recompensa = 0.0
        info = {}
        if terminado:
            self.resultado = self._resultado()
            info = dict(self.resultado)
            recompensa = self.resultado["ganancia"] / 1000.0
        return self._obs(), recompensa, terminado, False, info

    def _temporada_terminada(self):
        # La temporada termina cuando el cultivo se cosecha (AquaCrop marca crop_dead
        # o la fecha de cosecha). Tras la cosecha el riego ya no cambia el resultado,
        # asi que cortamos el episodio para ahorrar computo.
        c = self.model._init_cond
        return bool(getattr(c, "harvest_flag", False))

    def _resultado(self):
        fila = self.model._outputs.final_stats.iloc[0]
        rend = float(fila["Dry yield (tonne/ha)"])
        riego = float(fila["Seasonal irrigation (mm)"])
        return dict(anio=self.anio, rendimiento=rend, riego=riego, ganancia=ganancia(rend, riego))
