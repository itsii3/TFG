import optuna
import joblib
import math
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
from utils import transformacio_dades as td


class Personalitzacio:

    def __init__(self, dades):
        self.dades = dades
        self.intent_actual = 0
        self.millor_error_val = math.inf

    def _construir_model(self, parametres):
        model = Sequential()
        n_neurones = parametres['n_neurones']
        model.add(LSTM(n_neurones,
                       return_sequences=(parametres['n_capes'] > 1),
                       input_shape=(parametres['memoria'], self.dades['x'].shape[1])))
        model.add(Dropout(0.25))

        for capa in range(1, parametres['n_capes']):
            n_neurones = round(n_neurones / 2)
            model.add(LSTM(n_neurones, return_sequences=(capa < parametres['n_capes'] - 1)))
            model.add(Dropout(0.20))

        model.add(Dense(1))
        return model

    def _objectiu(self, intent):
        # selecció paràmetres de l'intent
        parametres = {
            'n_capes': intent.suggest_int('n_capes', 1, 4),
            'n_neurones': intent.suggest_int('n_neurones', 100, 300),
            'memoria': intent.suggest_int('memoria', 5, 20),
            'vel_apren': intent.suggest_float('vel_apren', 0.0005, 0.01, step=0.0005)
        }

        # definició i entrenament del model
        model = self._construir_model(parametres=parametres)
        model.compile(loss='mse', optimizer=optimizers.SGD(parametres['vel_apren']))
        dades_lstm = td.format_lstm(self.dades, memoria=parametres['memoria'])
        model.fit(dades_lstm['ent'],
                  validation_data=dades_lstm['val'],
                  epochs=300,
                  batch_size=64,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=30)],
                  shuffle=False,
                  verbose=0)

        # avaluació del model d'aquest intent
        error_val = model.evaluate(dades_lstm['val'])

        # actualització del millor model
        if error_val < self.millor_error_val:
            self.millor_error_val = error_val
            joblib.dump(model, 'model/millor_model')
            joblib.dump(parametres, 'model/millors_parametres')

        self.intent_actual = self.intent_actual + 1
        return error_val

    def començar_personalitzacio(self, n_intents):
        estudi = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        estudi.optimize(self._objectiu, n_trials=n_intents)

    def obtenir_prediccio_millor_model(self):
        model = joblib.load('model/millor_model')
        parametres = joblib.load('model/millors_parametres')
        model.compile(loss='mse', optimizer=optimizers.SGD(parametres['vel_apren']))
        dades_lstm = td.format_lstm(self.dades, memoria=parametres['memoria'])
        pred_total = model.predict(dades_lstm['total'])
        pred_test = model.predict(dades_lstm['test'])

        return {
            'pred_total': pred_total,
            'pred_test': pred_test
        }
