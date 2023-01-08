from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import seaborn as sns


def grafic_dades(dades, llegenda=[], titol='', guardar='', mida_figura=(10, 5)):
    fig = pyplot.figure(figsize=mida_figura)
    eix = fig.add_subplot(1, 1, 1)
    for dada in dades:
        eix.plot(dada)
    eix.set_title(titol)
    if len(llegenda) > 0:
        eix.legend(llegenda)
    if len(guardar) > 0:
        fig.savefig(guardar, bbox_inches='tight')
    fig.show()


def grafic_descomposicio_estacional(dades, titol='', guardar='', periode=52):
    descomposicio = sm.tsa.seasonal_decompose(dades, period=periode)
    tendencia = descomposicio.trend
    estacionalitat = descomposicio.seasonal
    residual = descomposicio.resid

    fig, axes = pyplot.subplots(4, 1, sharex='all', sharey='none')
    fig.set_figheight(8)
    fig.set_figwidth(15)
    axes[0].plot(dades)
    axes[0].set_ylabel(titol, fontsize=15)
    axes[1].plot(tendencia)
    axes[1].set_ylabel('Tendència', fontsize=12)
    axes[2].plot(estacionalitat)
    axes[2].set_ylabel('Estacionalitat', fontsize=12)
    axes[3].plot(residual)
    axes[3].set_ylabel('Residual', fontsize=12)

    if len(guardar) > 0:
        fig.savefig(guardar, bbox_inches='tight')


def matriu_correlacio(dades, guardar='', col_guardar=[], mida_figura=20):
    if len(col_guardar) > 0:
        columnes = [dades.columns[col] for col in col_guardar]
        dades = dades.loc[:, columnes]
    corr = dades.corr()
    grafic = sns.heatmap(corr, vmax=1, square=True, annot=True, cmap='cubehelix')

    fig = grafic.get_figure()
    fig.set_figheight(mida_figura)
    fig.set_figwidth(mida_figura)
    if len(guardar) > 0:
        fig.savefig(guardar, bbox_inches='tight')


def matriu_dispersio(dades, guardar='', col_guardar=[], mida_figura=20):
    if len(col_guardar) > 0:
        columnes = [dades.columns[col] for col in col_guardar]
        dades = dades.loc[:, columnes]
    dispersio = scatter_matrix(dades, figsize=(mida_figura, mida_figura))
    if len(guardar) > 0:
        pyplot.savefig(guardar)
