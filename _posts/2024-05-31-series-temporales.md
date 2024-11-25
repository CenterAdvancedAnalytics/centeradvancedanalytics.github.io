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
