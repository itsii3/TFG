import datetime
import pandas as pd


_format_dates = '%Y-%m-%d'


def omplir_dies_faltants(dades):
    inici, fi = dades.index[0], dades.index[-1]
    rang = [dia.date().strftime(_format_dates) for dia in pd.date_range(inici, fi)]
    return pd.DataFrame(dades[dades.columns[0]], index=rang)


def dies_entre_setmana(inici, fi):
    rang = [dia.date() for dia in pd.date_range(inici, fi)]
    rang_entre_setmana = list(filter(lambda dia: dia.weekday() < 5, rang))
    return [dia.strftime(_format_dates) for dia in rang_entre_setmana]


def percentatge_dades_faltants(dades):
    inici, fi = dades.index[0], dades.index[-1]
    rang = dies_entre_setmana(inici=inici, fi=fi)
    return 1 - len(dades.index)/len(rang)


def es_dilluns(dia):
    return datetime.datetime.strptime(dia, _format_dates).weekday() == 0
