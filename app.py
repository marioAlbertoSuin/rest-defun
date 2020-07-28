import flask
from flask import request
import pandas as pd
from datetime import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as mplt
import seaborn as sbn
import statsmodels.api as sm
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    fecha = request.args['fecha']
   
    #------------------------------------><-----------------------------------------------------
    #cargar dataset
    DF_2017 = pd.read_csv('EDF_2017_V2.csv', sep=';',encoding='windows-1250')
    DF_2016 = pd.read_csv('EDF_2016.csv', sep=',',encoding='windows-1250')
    DF_2018 = pd.read_csv('2018.csv', sep=';',encoding='windows-1250')
    #funcion limpiar datos sin informacion
    def datos_sin_informacion(DF):
        if (DF.sexo.dtypes == object):DF.sexo = DF.sexo.replace({"Sin información": 'Indeterminado'})
        if (DF.hij_viv.dtypes == object):DF.hij_viv = DF.hij_viv.replace({"Sin información": '0'})
        if (DF.hij_vivm.dtypes == object):DF.hij_vivm = DF.hij_vivm.replace({"Sin información": '0'})
        if (DF.hij_nacm.dtypes == object):DF.hij_vivm = DF.hij_nacm.replace({"Sin información": '0'})
        if (DF.con_pren.dtypes == object):DF.con_pren = DF.con_pren.replace({"Sin información": '0'})
        return DF
    #funcion para definir como date
    def cambiar_tipo_variable_tiempo (df):
        DF['fecha_insc'] = pd.to_datetime(DF.fecha_insc)
        DF['fecha_fall'] = pd.to_datetime(DF.fecha_fall)
        DF['fecha_mad'] = pd.to_datetime(DF.fecha_insc,'yyyy/mm/dd')
        return DF
    #Compacta los documentos
    DF = pd.concat([DF_2016, DF_2017, DF_2018], axis=0)

    #limpiar datos y cambiar variables Date
    DF = cambiar_tipo_variable_tiempo(datos_sin_informacion(DF))

    #nuevo dataframe con las fechas de inscripcion y el sexo de la defuncion
    fecha_insc_sexo=DF.iloc[:, [6,7]]
    #crear nuevas variables que remplaza la variable categorica
    fecha_insc_sexo = pd.get_dummies(fecha_insc_sexo, columns=['sexo'])

    #Agrupa las fechas y los totales de cada defuncion 
    total_df_por_sexo = fecha_insc_sexo.groupby(fecha_insc_sexo['fecha_insc']).agg(['sum'])
    total_df = DF.fecha_insc.groupby(fecha_insc_sexo['fecha_insc']).agg(['count'])

    #################   Pronosticando la serie con SARIMAX   ###############
    # Modelo ARIMA sobre el valor de cierre de la acción.
    modelo_sexo_hombre = sm.tsa.statespace.SARIMAX(total_df_por_sexo['sexo_Hombre'], order=(0, 1, 0),)
    modelo_sexo_Indeterminado = sm.tsa.statespace.SARIMAX(total_df_por_sexo['sexo_Indeterminado'], order=(0, 1, 0))
    modelo_sexo_Mujer = sm.tsa.statespace.SARIMAX(total_df_por_sexo['sexo_Mujer'], order=(0, 1, 0),)
    modelo_total_df = sm.tsa.statespace.SARIMAX(total_df['count'], order=(0, 1, 0),)

    #print(total_df_por_sexo['sexo_Hombre'].iloc[1:])
    resultados = modelo_sexo_hombre.fit()
    resultados_indeterminado = modelo_sexo_Indeterminado.fit()
    resultados_Mujer = modelo_sexo_Mujer.fit()
    resultados_total_df = modelo_total_df.fit()

    SARIMAX_predict = round(resultados.predict())
    SARIMAX_predict_indeterminado = round(resultados_indeterminado.predict())
    SARIMAX_predict_mujer = round(resultados_Mujer.predict())
    SARIMAX_predict_total = round(resultados_total_df.predict())
    idx = pd.date_range('2018-6-01', '2020-12-31')

    SARIMAX_predict = pd.DataFrame(list(zip(list(idx),list(SARIMAX_predict))),
    columns=['Date','Defunciones_Masculino']).set_index('Date')

    SARIMAX_predict_indeterminado = pd.DataFrame(list(zip(list(idx),list(SARIMAX_predict_indeterminado))),
    columns=['Date','Defunciones_Indeterminado']).set_index('Date')

    SARIMAX_predict_mujer = pd.DataFrame(list(zip(list(idx),list(SARIMAX_predict_mujer))),
    columns=['Date','Defunciones_Mujer']).set_index('Date')

    SARIMAX_predict_total = pd.DataFrame(list(zip(list(idx),list(SARIMAX_predict_total))),
    columns=['Date','Defunciones']).set_index('Date')


    # Valores al cierre de cada dia
    df_ma_d = SARIMAX_predict
    df_ind_d = SARIMAX_predict_indeterminado
    df_muj_d = SARIMAX_predict_mujer
    df_total_d = SARIMAX_predict_total
    #print("Predicciones sexo Hombre\n",df_ma_d)
    #print("Predicciones sexo indeterminado\n",df_ind_d)
    #print("Predicciones sexo Mujer\n",df_muj_d)
    #print("Predicciones defunciones\n",df_total_d)

    # Valores al cierre de cada mes
    df_ma_m = SARIMAX_predict.resample("M").sum()
    df_ind_m = SARIMAX_predict_indeterminado.resample("M").sum()
    df_muj_m = SARIMAX_predict_mujer.resample("M").sum()
    df_total_m = SARIMAX_predict_total.resample("M").sum()

    #print(df_ma_m)
    #print(df_ind_m)
    #print(df_muj_m)
    #print(df_total_m)

    # Valores al cierre de cada año
    df_ma_y = SARIMAX_predict.resample("Y").sum()
    df_ind_y = SARIMAX_predict_indeterminado.resample("Y").sum()
    df_muj_y = SARIMAX_predict_mujer.resample("Y").sum()
    df_total_y = SARIMAX_predict_total.resample("Y").sum()

    #print(df_ma_y)
    #print(df_ind_y)
    #print(df_muj_y)
    #print(df_total_y)

    import json
    def pasar_json(df):
        df = df.to_json(orient='table')
        df = json.loads(df)
        df = df['data']
        return df

    ################# json #############
    ###dia#####
    df_ma_d = pasar_json(df_ma_d)
    df_ind_d = pasar_json(df_ind_d)
    df_muj_d = pasar_json(df_muj_d)
    df_total_d = pasar_json(df_total_d)
    dia={"datos-mujer":df_muj_d,"datos-masculino":df_ma_d,"datos-indeterminado":df_ind_d,"total":df_total_d}
    ##mes####
    df_ma_m = pasar_json(df_ma_m)
    df_ind_m = pasar_json(df_ind_m)
    df_muj_m = pasar_json(df_muj_m)
    df_total_m = pasar_json(df_total_m)
    mes={"datos-mujer":df_muj_m,"datos-masculino":df_ma_m,"datos-indeterminado":df_ind_m,"total":df_total_m}
    ###anio####
    df_ma_y = pasar_json(df_ma_y)
    df_ind_y = pasar_json(df_ind_y)
    df_muj_y = pasar_json(df_muj_y)
    df_total_y = pasar_json(df_total_y)
    ano={"datos-mujer":df_muj_y,"datos-masculino":df_ma_y,"datos-indeterminado":df_ind_y,"total":df_total_y}
    if(fecha=="anio"):return ano
    if(fecha=="mes"):return mes
    if(fecha=="dia"):return dia



#------------------------------------><-----------------------------------------------------
    

