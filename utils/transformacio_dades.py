import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from utils import utilitats_dates as ud


def interpolacio_lineal(dades):
    return ud.omplir_dies_faltants(dades).interpolate(method='linear', axis=0)


def interpolacio_enrere(dades):
    return ud.omplir_dies_faltants(dades).bfill(axis='rows')


def format_lstm(dades, memoria):
    st_total = TimeseriesGenerator(dades['x'], dades['y'], memoria, sampling_rate=1, batch_size=1)
    st_ent = TimeseriesGenerator(dades['x_ent'], dades['y_ent'], memoria, sampling_rate=1, batch_size=1)
    st_val = TimeseriesGenerator(dades['x_val'], dades['y_val'], memoria, sampling_rate=1, batch_size=1)
    st_test = TimeseriesGenerator(dades['x_test'], dades['y_test'], memoria, sampling_rate=1, batch_size=1)
    return {
        'total': st_total,
        'ent': st_ent,
        'val': st_val,
        'test': st_test
    }
