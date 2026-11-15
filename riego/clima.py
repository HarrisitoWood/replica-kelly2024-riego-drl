"""
Carga del clima sintetico (LARS-WG) de Champion, Nebraska y calculo de ETo.

Replica la funcion `calc_eto_faopm` del repositorio original aquacrop-gym
(Kelly et al., 2024). El archivo CPWG_100.dat contiene los primeros 100 anos
del clima generado con LARS-WG que publicaron los autores:
  anos 1-70  -> entrenamiento
  anos 71-100 -> validacion
"""
from pathlib import Path

import numpy as np
import pandas as pd
from aquacropeto import (
    avp_from_tmin, atm_pressure, cs_rad, deg2rad, delta_svp, et_rad,
    fao56_penman_monteith, inv_rel_dist_earth_sun, mean_svp, net_in_sol_rad,
    net_out_lw_rad, net_rad, psy_const, sol_dec, sunset_hour_angle,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "CPWG_100.dat"

# Estacion Champion, Nebraska (cabecera del archivo LARS-WG original)
LATITUD = 40.4
ALTITUD = 1072
ANIO_CALENDARIO = 1995  # ano "ficticio" que usa el codigo original


def cargar_clima(path=DATA, year=ANIO_CALENDARIO, latitude=LATITUD, altitude=ALTITUD):
    """Devuelve un DataFrame con columnas simyear, MinTemp, MaxTemp,
    Precipitation, ReferenceET, Date (formato que pide AquaCrop-OSPy)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = ["simyear", "jday", "minTemp", "maxTemp", "precip", "rad"]
    df["date"] = pd.to_datetime(f"{year - 1}/12/31") + pd.to_timedelta(df.jday, unit="D")

    lat = deg2rad(latitude)
    sd = sol_dec(df.jday)
    ext_rad = et_rad(lat, sd, sunset_hour_angle(lat, sd), inv_rel_dist_earth_sun(df.jday))
    net_sw = net_in_sol_rad(df.rad)
    net_lw = net_out_lw_rad(df.minTemp, df.maxTemp, df.rad, cs_rad(altitude, ext_rad),
                            avp_from_tmin(df.minTemp))
    rn = net_rad(net_sw, net_lw)
    tmean = (df.minTemp + df.maxTemp) * 0.5
    eto = fao56_penman_monteith(
        rn, tmean + 273, 2, mean_svp(df.minTemp, df.maxTemp), avp_from_tmin(df.minTemp),
        delta_svp(tmean), psy_const(atm_pressure(altitude)),
    )
    df["eto"] = np.clip(eto, 0.1, None)
    out = df[["simyear", "minTemp", "maxTemp", "precip", "eto", "date"]].copy()
    out.columns = ["simyear", "MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]
    return out


def separar_por_anio(gendf):
    """Diccionario {anio_sintetico: DataFrame} para no filtrar en cada episodio."""
    return {int(y): g.drop(columns="simyear").reset_index(drop=True)
            for y, g in gendf.groupby("simyear")}
