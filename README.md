#  Código utilizado en el proyecto de investigacción de detección de hipoxia fetal con redes neuronales.
La hipoxia fetal es una condición médica grave en la que el feto no recibe suficiente oxígeno durante el embarazo, lo que puede provocar daños cerebrales y otros problemas de salud. La detección temprana de la hipoxia fetal es fundamental para tomar medidas para proteger la salud del feto.
En los últimos años, el uso de técnicas de aprendizaje profundo, específicamente redes neuronales profundas, se ha convertido en un enfoque prometedor para la detección de hipoxia fetal a partir de señales fisiológicas. Las redes neuronales profundas han demostrado su eficacia en la identificación de patrones complejos en señales biomédicas, lo que las hace ideales para la detección de hipoxia fetal a partir de señales de frecuencia cardíaca fetal y contracciones uterinas.
En este contexto, el objetivo de este repositorio es proporcionar un conjunto de funciones útiles que han sido utilizadas en el desarrollo de un proyecto de investigación sobre la detección de hipoxia fetal mediante redes neuronales profundas.  Dichas funciones se pueden encontrar  en el archivo  **codigo.py**. Además, se proporcionan otros archivos con código para el desarrollo de modelos y gráficos, ilustrando el funcionamiento de las funciones mencionadas.
Espero que este repositorio sea de ayuda para la comunidad científica  y pueda contribuir al avance en dicho problema,  así como puedan ser de gran utilidad para otros investigadores y profesionales del campo de la medicina y la ingeniería biomédica  para el procesamiento y análisis de señales.

### Descripción de las funciones.
> **Preprocesado de señales**. <br>
>  El preprocesamiento de señales es una tarea fundamental en la detección de hipoxia fetal mediante redes neuronales profundas. El objetivo de esta etapa es preparar las señales de entrada para que puedan ser utilizadas por los modelos de aprendizaje profundo, eliminando ruido y artefactos y resaltando las características relevantes de la señal. Estas funciones son aplicadas a las señales  de CTG,  contenidas en archivos .dat.  Cabe destacar que las funciones han sido creadas con la idea de ejecutarlas  en el orden que se muestran. <br>
>  - clean_infal(): función  que elimina los valores nulos de los extremos de las señales de FHR y UC.  Sin embargo, solo suprime  aquellos en los que se cumpla que tanto la posición en la señal de FHR como en la de UC poseen valor nulo.  Solo toma de entrada  el número  de identificación de la señal, el cual esta contenido en el nombre del archivo.  Como salida se obtiene un array que contiene los valores de las señales procesadas. <br>
>  - gap_clean(): función que elimina fragmentos de valores nulos en el interior de la señal de un  cierto tamaño. Al igual que antes, el  fragmento debe ser  de valores nulos en  las mismas posiciones de ambas señales. Toma como  entrada la señal y el tamaño del fragmento . Como salida se obtiene  un array que contiene los valores de las señales ya procesadas. <br>
>  - zeros_to_nan(): esta función  tiene como entrada la señal y además ciertos umbrales  a partir de los cuales  los valores de las señales son considerados como ruido.  Siguiendo esto, la función convierte los valos nulos restantes de las señales en  valores no numéricos (NaN), así como  todos aquellos que superen o sean inferiores a los umbrales establecidos para cada señal. Finalmente, se obtiene como resultado un a rray con los valores de las señales limpias. <br>

>**Etiquetado**. <br>
>  Dado que el problema abordado  es de clasificación binaria, es necesario contar con etiquetas que permitan clasificar cada señal como normal o hipóxica. Por este motivo se desarolla una función que devuelva  la etiqueta de cada una de las señales en función del ph del cordón umbilical, dato que puede encontrarsese en los archivos .hea de cada señal. 
> - target(): función que toma de entrada  el valor del ph umbilical y  un umbral, a partir del cual se establecerán  ambas clases. De esta manera, las señales cuyo valor de ph asociado sea inferior  al umbral establecido serán clasificadas como hipóxicas y etiquetadas como consecuencia con un 1. Por el contrario, si el ph asociado es superior al umbral serán clasificadas como normales y etiquetadas con un 0. Como salida se obtiene un array con las etiquetas de las señales y el número de señales normles y señales hipóxicas.

>**Arquitecturas de los modelos**.<br>
> Se presentan en este apartado las funciones  que permiten la construcción y entrenamiento de modelos basados en dos arquitecturas populares: redes convolucionales y redes LSTM.
>  - build_cnn_model():  función utilizada para construir una red  neuronal convolucional. Recibe como argumentos la cantidad de unidades (neuronas) en la capa convolucional, el tamaño del kernel (ventana de filtrado) y el valor de regularización para el kernel. En la función, primero se crea una instancia del modelo Sequential de Keras, que se utiliza para apilar capas de forma secuencial. Luego, se agrega una capa convolucional con activación ReLU y regularización L2, seguida de una capa de maxpooling para reducir la dimensionalidad de la salida de la capa convolucional.  Después, se agrega una capa de flatten para aplanar la salida y pasarla a una capa totalmente conectada con una sola neurona y activación sigmoide, lo que convierte la salida en una probabilidad entre 0 y 1. Finalmente, el modelo se compila con la  función de  pérdida de entropía cruzada binaria, el optimizador Adam y se utiliza la precisión y el área bajo la curva ROC (AUC) como métricas para evaluar su desempeño. <br>
>  - build_LSTM_model(): función utilizada para crear un modelo de red neuronal basado en una arquitectura LSTM. En esta función, se especifica el número de unidades (neuronas) en la capa LSTM y se añade una capa de dropout para evitar el sobreajuste. Además, se aplica una normalización por lotes para normalizar la salida de la capa LSTM y se agrega una capa totalmente conectada con una activación sigmoide para la clasificación binaria. Finalmente, al igual que en la función anterior, Finalmente, se compila el modelo especificando la función de pérdida, el optimizador y las métricas de evaluación. <br>
>  - build_cnn_model_2(): función utilizada para construir una CNN profunda. A diferencia de la función *build_cnn_model*, se añade una capa convolucional y una capa de Maxpooling adicional a la red, además de una capa de normalización por lotes, lo que puede mejorar el rendimiento del modelo. Como argumentos de entrada recibe el número de filtros, el tamaño del filtro y el factor de regularización de cada capa convolucional, más el tamaño del filtro de la primera capa de Maxpooling. <br>
>  - build_LSTM_model_2(): variante de la función *build_LSTM_model*, formada por dos capas LSTM, que toma como parámetros de entrada el número de unidades de cada capa LSTM y valor de la capa Dropout.

>**Transformada de Fourier de las señales**. <br>
>La transformada de Fourier es una herramienta matemática que se utiliza para descomponer una señal en sus componentes frecuenciales. Alimentar los modelos con la FFT de las señales en vez de con las señales temporales directamente puede ser beneficioso, ya que reduce la dimensionalidad, y como consecuencia hace más eficiente el proceso de entrenamiento. Además, puede ser de gran utilidad para eliminar el ruido de la señal. <br>
>- fourier(): función que recibe como entrada una matriz que contiene señales de dos canales (FHR y UC), calcula la transformada de Fourier para cada fila de la matriz en cada canal, normaliza los datos obtenidos y almacena las frecuencias positivas (hasta la frecuencia de Nyquist). <br>

>**Gráfica de la curva ROC**. <br>
>La curva ROC representa la relación entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR) para diferentes umbrales de clasificación. En otras palabras, la curva ROC muestra cómo varía la sensibilidad del modelo (TPR) en función de la especificidad (FPR). La curva ROC se representa gráficamente con el eje Y representando el TPR y el eje X el FPR. Idealmente, un modelo de clasificación binaria perfecto tendría un área bajo la curva (AUC) igual a 1, lo que significa que tiene una tasa de verdaderos positivos del 100% y una tasa de falsos positivos del 0%. Por otro lado, un modelo aleatorio tendría un AUC igual a 0,5, lo que significa que no hay diferencia entre las tasas de verdaderos positivos y falsos positivos. En general, cuanto mayor sea el AUC de una curva ROC, mejor será el modelo para predecir la clase positiva. <br>
> - plot_roc(): función que genera un curva ROC a partir de las predicciones de un modelo y las etiquetas originales. La función recibe como argumentos un nombre que indica una determinada  partición del conjunto de datos de un modelo, un array con las etiquetas verdaderas de las señales, un array con las predicciones del modelo y por último, argumentos adicionales que pueden ser utilizados para personalizar la línea de la curva ROC, como el color o el estilo.
> 
### Entrenamiento de modelos
El archivo modelos.ipynb contiene un ejemplo del código desarrollado para el entrenamiento de los diferentes modelos. En este caso se presenta el modelo CNN entrenado con la transformada de Fourier de las señales de CTG. Como se observa, se define una estrategia de validación cruzada con 15 divisiones, que permite obtener una medida mas generalizada del rencimiento real del modelo, pues al entrenar un modelo con una sola partición de datos, es posible que el modelo se sobreajuste o subajuste a los patrones específicos de esa partición. Además, se utiliza una búsqueda en cuadrícula de los hiperparámetros para optimizar el modelo CNN, y se guarda el modelo con los mejores parámetros. Finalmente se imprime las curvas ROCs referentes a cada partición y la media de las métricas.

### Código para la obtención de gráficas de evolución de métricas y curvas ROC
Por último, el archivo Graficas_TFM.ipynb contiene el código desarrollado en el trabajo para la obtención de gráficas. En un primer lugar se presentan las gráficas de evolución de las métricas utilizadas en los modelos entrenados con la partición con la que se ha obtenido mejor rendimiento. Para ello, se entrena un modelo, con la partición óptima y con los hiperparámetros que mejor resultado producen. Se entrena un modelo concreto para cada una de las arquitecturas desarrolladas en el trabajo. Además, se muestra el código empleado para la obtención de la curva ROC media obtenida con la media del rendimiento obtenido por cada una de los modelos desarollados. Para ello, se cargan las predicciones de los modelos, se calcula la tasa de falsos positivos y negativos del modelo sobre cada una de las 15 particiones y se calcula la media y la desviación típica, obteniendo finalmente la curva ROC media con con su desviacón típica, representando así un resultado más general. Por último, se muestra el código desarrolado para la impresión de una gráfica resumen, que muestra la media de cada modelo, junto el mejor y el peor rendimiento obtenido.
