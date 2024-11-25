## Series Temporales en el mundo de la ciencia de datos
*¿Te gustaría poder predecir el futuro? Si hubieras invertido un dólar en acciones de Apple en el año 2008, hoy día esa inversión se hubiera convertido en 187 dólares. Algo similar hubiera ocurrido si hubieses invertido ese dinero en cualquier otra de las Big Tech. Sin embargo, existen múltiples alternativas de inversión en todo momento, e identificar cuál de ellas generará estos niveles de retorno es imposible porque no existe una bola de cristal que nos prediga el futuro. ¿O sí?*

### Series de tiempo
Vivimos en un mundo volátil, incierto, complejo y ambiguo, en el que tomamos decisiones que dependen de eventos futuros que desconocemos. En este contexto, el análisis de series temporales surge como una disciplina esencial, que ofrece respuestas valiosas sobre tendencias y patrones de datos a lo largo del tiempo, y nos puede ayudar así a predecir el futuro las variables que nos interesa

Una serie de tiempo es una secuencia de variables aleatorias indexadas por el tiempo. Los datos pueden abarcar una amplia variedad de campos, siendo algunos, por ejemplo precios, demanda de energía, tráfico de una página web, etc. Los métodos de análisis de series temporales son, fundamentalmente, maneras de descubrir patrones dentro de esta secuencia de variables aleatorias ordenadas. Por ejemplo, tendencias, ciclos y estacionalidades.

Consideremos la evolución histórica de precios de acciones de Apple (“APPL”), en la librería “yfinance” de Python. Esta librería permite acceder de manera sencilla a datos financieros como precios de acciones, volúmenes de transacción, y otros indicadores. En este caso en particular, el periodo de tiempo que determinaremos estará comprendido entre enero de 2019 y finales de mayo del 2024 en frecuencia diaria.

```python
apple_data = yf.download('AAPL', start='2019-01-01', end='2024-05-26', progress=False)


appled_ata.reset_index(inplace=True)
```
Podemos observar la serie histórica de precios de cierre de Apple utilizando la librería “plotly” de la siguiente manera:
```python
fig_close = px.line(apple_data,
                   x="Date",
                   y="Close",
                   labels = {
                       "Date": "Fecha",
                       "Close": "Precio de Cierre"
                   },
                   title="Evolución Histórica de los Precios de Apple - Últimos 5 Años"
                   )


fig_close.show()
```

![APPL Series](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-31-series-temporales/appl_5yrs.png "APPL Series")

Al dataframe “apple_data” le añadimos una variable llamada “Retornos”, calculada a partir de la variación diaria del precio de cierre. Esta variable nos permite conocer las rentabilidades diarias de Apple a lo largo del tiempo. Podemos hacer inspección de esta variable con un histograma.

```python
apple_data["Returns"] = apple_data["Close"].pct_change()

fig_hist = px.histogram(apple_data,
                        x='Returns',
                        labels={
                          'Returns': 'Retornos',
                        },
                        color_discrete_sequence=['goldenrod'],
                        title='Distribución de Retornos Diarios: AAPL vs SP500'
                        )
fig_hist.show()
```

### Prophet: Un modelo innovador en el mundo de las predicciones temporales
Entre las técnicas más recientes se encuentra el modelo desarrollado por Meta en el año 2017, conocido como “*Prophet*”. Este modelo ofrece mejoras en la capacidad predictiva al descomponer las series temporales en sus componentes de tendencia, estacionalidad, y efectos vacacionales. Gracias a esta descomposición, Prophet es capaz de capturar patrones complejos y variaciones estacionales, proporcionando predicciones precisas y detalladas.

Prophet es especialmente útil cuando intentamos estimar series que presentan cambios abruptos o pueden verse afectadas por eventos externos como lo son las promociones o la volatilidad de los mercados. Además, gracias a su facilidad de uso y flexibilidad es relativamente sencillo llevar a cabo una estimación sin conocimientos profundos en estadística. Todo esto lo convierte en una herramienta valiosa para una amplia gama de aplicaciones como pronóstico de ventas, planificación de inventarios, métricas de redes sociales, etc.

Realizaremos una estimación Prophet aplicada a la serie de Apple y evaluaremos su desempeño comparando los valores reales con los estimados, esto se logra dividiendo el dataset en conjuntos de train y test, siendo los últimos tres días el conjunto de test. Como métrica de desempeño utilizamos el **MAPE** (Error Porcentual Absoluto Medio), el cual calcula la diferencia porcentual entre los valores reales y los estimados.

```python
apple_prophet = apple_data[["Date", "Close"]]
apple_prophet = apple_prophet.rename(columns = {"Date": "ds", "Close": "y"})


train = apple_prophet[:-3]
test = apple_prophet[-3:]


m = Prophet()


m.fit(train)


future = m.make_future_dataframe(periods=3,
                                 freq="D"
                                 )


forecast = m.predict(future)


forecast_data = forecast[["ds", "yhat"]].tail(3)


compared_data = pd.merge(test, forecast_data, on="ds")


for index, row in compared_data.iterrows():
    mape = mean_absolute_percentage_error([row["y"]], [row["yhat"]]) * 100
    compared_data.at[index, "MAPE"] = mape




compared_data["MAPE"] = mape
```

La comparación de los valores reales y estimados en el periodo de prueba se muestra en la siguiente tabla junto con el MAPE asociado a cada estimación:

![Prophet Table](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-31-series-temporales/table1.png "Prophet Table")

Los resultados evidencian un desempeño en promedio del modelo Prophet del 6% aproximadamente.

### El Poder de los modelos ARIMA en el Análisis de Series Temporales
Una metodología ampliamente utilizada a lo largo del tiempo es la estimación con modelo ARIMA (Autoregressive Integrated Moving Average), herramienta poderosa para modelar datos temporales que presentan autocorrelación. Estos modelos se componen de tres partes principales: el componente autoregresivo (AR), el componente de media móvil (MA) y la diferenciación integrada (I). El concepto de modelos ARIMA fue popularizado por George Box y Gwilym Jenkins en su libro “Time Series Analysis: Forecasting and Control”, publicado en 1970. Estos modelos son capaces de capturar una amplia gama de patrones temporales, incluidas tendencias, estacionalidades y comportamientos aleatorios. El enfoque ARIMA ha sido ampliamente utilizado en el mundo de la economía y finanzas, el control de calidad, análisis demográfico, entre otros.

Si realizamos una estimación de precios para Apple utilizando un modelo ARIMA (1,1,0) obtenemos resultados interesantes. En primer lugar, una vez definido nuestros periodos de entrenamiento y test, llamamos a la función ARIMA de la librería statsmodels de Python, indicamos que deseamos 3 periodos a estimar y observamos los resultados en compared_data_arima.

```python
apple_arima = apple_data[["Date", "Close"]]
apple_arima = apple_prophet.rename(columns = {"Date": "ds", "Close": "y"})


train = apple_arima[:-3]
test = apple_arima[-3:]


arima_model = sm.tsa.ARIMA(train["y"], order=(1,1,0))


arima_result = arima_model.fit()


forecast_arima = arima_result.forecast(steps=3)


forecast_arima_df = pd.DataFrame(forecast_arima, columns=["predicted_mean"])


compared_data_arima = pd.concat([test, forecast_arima], axis = 1)
compared_data_arima.reset_index(drop=True, inplace=True)


for index, row in compared_data_arima.iterrows():
    mape = mean_absolute_percentage_error([row["y"]], [row["predicted_mean"]])*100
    compared_data_arima.at[index, "MAPE"] = mape
```

Como se muestra en la tabla siguiente, el MAPE del modelo ARIMA(1,1,0) es menor que el del modelo Prophet. Esto quiere decir que las predicciones del modelo ARIMA se acercan más a los valores reales en comparación que los valores de Prophet.

![ARIMA Table](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-31-series-temporales/table2.png "ARIMA Table")

### Estimador XGBoost: Machine Learning aplicado al mundo de las Series Temporales.
Aparte de los modelos Prophet y ARIMA, otros enfoques avanzados incluyen algoritmos de inteligencia artificial como XGBoost (Extreme Gradient Boosting) el cual ha demostrado ser un estimador eficaz debido a su capacidad de manejar datos heterogéneos y relaciones complejas entre variables. Para estimaciones de series temporales, el modelo puede incorporar diversas características que influyen en la variable objetivo. Cada característica aporta información adicional, ayudando al modelo a entender mejor las variaciones y tendencias en los datos. Por ejemplo, el volúmen de transacciones de una acción puede aportar información sobre el nivel de liquidez en el mercado, mientras que el día de la semana puede capturar patrones de comportamiento de los inversores.

Realizaremos una estimación XGBoost en el precio de cierre histórico de Apple, tomando como características influyentes el volumen de transacciones y variables temporales como el año, el mes, el día, entre otros.

```python
#seleccionamos las caracteristicas
apple_data["Year"] = apple_data["Date"].dt.year
apple_data["Month"] = apple_data["Date"].dt.month
apple_data["Day"] = apple_data["Date"].dt.day
apple_data["DayOfWeek"] = apple_data["Date"].dt.dayofweek


apple_data.dropna(inplace=True)


x = apple_data[["Year", "Month", "Day", "DayOfWeek", "Volume"]]
y = apple_data["Close"]


#definimos los conjuntos de train y test
train_size = len(apple_data)-3
x_train, x_test = x.iloc[:train_size], x.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


#utilizamos el modelo
model = XGBRegressor(objective="reg:squarederror",
                     n_estimators=100,
                     learning_rate=0.1
                     )
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


#generamos un dataframe ara comparar resultados
compared_data_xgboost = pd.DataFrame({
    "Date":apple_data["Date"].iloc[train_size:],
    "Actual": y_test.values,
    "Predicted": y_pred
})


for index, row in compared_data_xgboost.iterrows():
    mape = mean_absolute_percentage_error([row["Actual"]], [row["Predicted"]]) * 100
    compared_data_xgboost.at[index, "MAPE"] = mape
```
En promedio, el margen de error de las predicciones del estimador XGBoost es de 1.19%. Específicamente, las estimaciones del modelo se ven en la siguiente tabla:

![XGBoost Table](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-31-series-temporales/table3.png "XGBoost Table")

### Análisis comparativo y recomendaciones
Hemos llevado a cabo estimaciones para el precio de cierre de acciones de Apple utilizando tres estimadores diferentes: Prophet, ARIMA y XGBoost. La elección de cada modelo depende de los objetivos específicos y el contexto del análisis. Cada metodología ofrece ventajas y desventajas particulares que deben ser consideradas al seleccionar el modelo adecuado.

![Comparative Table](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-31-series-temporales/table4.png "Comparative Table")


