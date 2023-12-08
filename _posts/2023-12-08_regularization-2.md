![the data crafery shop.jpg](logo.png)

#### Módulo 2: Machine Learning Aplicado

# 6. Regularización

#### Contenidos:
* [6.1 El problema del overfitting](#1)
    * [6.1.1 Overfitting en un problema de regresión](#6.1.1-Overfitting-en-un-problema-de-regresión)
    * [6.1.2 Overfitting en un problema de clasificación](#6.1.2-Overfitting-en-un-problema-de-clasificación)
    * [6.1.3 Posibles soluciones al overfitting](#6.1.3-Posibles-soluciones-al-overfitting)
* [6.2 Regularización en Regresión Lineal](#2)
    * [6.2.1 Modificación de la función de coste](#6.2.1-Modificación-de-la-función-de-coste)
    * [6.2.2 Solución directa](#6.2.2-Solución-directa)
    * [6.2.3 Descenso por gradiente](#6.2.3-Descenso-por-gradiente)
* [6.3 Regularización en Regresión Logística](#3)
    * [6.3.1 Modificación de la función de coste](#6.3.1-Modificación-de-la-función-de-coste)
    * [6.3.2 Descenso por gradiente](#6.3.2-Descenso-por-gradiente)
* [6.4 Regularización en scikit-learn](#4)
    * [6.4.1 Regresión lineal (SGDRegressor y Ridge)](#6.4.1-Regresión-lineal-(SGDRegressor-y-Ridge))
    * [6.4.2 Regresión logística (SGDClassifier y LogisticRegression)](#6.4.2-Regresión-logística-(SGDClassifier-y-LogisticRegression))
* [6.5 Resumen y conclusiones](#5)


En este tema vamos a ver uno de los problemas más comunes que nos puede surgir cuando entrenamos un modelo de Machine Learning: el *overfitting*, sobre-ajuste o sobre-aprendizaje. Veremos en qué consiste este problema y qué medidas podemos tomar para solucionarlo. Nos centraremos en los métodos vistos hasta ahora: la regresión lineal y la regresión logística. 

## 6.1. El problema del overfitting  <a class="anchor" id="1"></a>

El overfitting, también conocido como sobre-ajuste o sobre-entrenamiento, es el hecho de que un modelo de aprendizaje automático se ajuste en exceso a los datos con los que ha sido entrenado, perdiendo así capacidad de predecir resultados con datos nuevos. 

No olvidemos que el objetivo de un sistema de Machine Learning es realizar **predicciones**, es decir, demostrar buen comportamiento con ejemplos del problema no vistos durante el entrenamiento: tener capacidad de **generalización**. 

En esta sección vamos a visualizar el efecto del overfitting en un problema de regresión y otro de clasificación.

### 6.1.1 Overfitting en un problema de regresión

Para visualizar el overfitting en un problema de regresión, nos vamos a basar en el problema de estimar el precio de una vivienda en función de sus metros cuadrados. 

Utilizamos los siguientes datos:


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gen_house_prices_nonlinear(n, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Para poder replicar los resultados
        
    # Generamos aleatoriamente una serie de ejemplos (x, y) - (tamaño, precio)
    x = np.random.randint(60, 200, n).reshape(-1, 1) # tamaños
    coef = [-1.54414014e-06,  9.76700044e-04, -2.33920721e-01,  2.57079894e+01, -717.628339190677]
    y = coef[0]*x**4 + coef[1]*x**3 + coef[2]*x**2 + coef[3]*x + coef[4] + np.random.randint(-15, 15, n).reshape(-1, 1) # precio en euros
  
    df = pd.DataFrame(np.hstack((x, y)), columns=[r'$m^2$', 'Precio (miles €)'])
    
    return x, y, df
    
x, y, df = gen_house_prices_nonlinear(10, seed=2)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$m^2$</th>
      <th>Precio (miles €)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>268.854582</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.0</td>
      <td>382.576042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>82.0</td>
      <td>299.251078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103.0</td>
      <td>347.101816</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.0</td>
      <td>369.906348</td>
    </tr>
  </tbody>
</table>
</div>



Como solo tenemos una variable, vamos a representar gráficamente los datos. 


```python
# Vamos a hacer una función que nos simplifique los plots
def plot_data_reg(x, y, labelx, labely, xlim=None, ylim=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, '.r', markersize=12)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax
_ = plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)')
```


    
![png](output_10_0.png)
    


Si ajustamos varios modelos de regresión polinómica a los datos, obtenemos diferentes soluciones en función del grado de polinomio que escojamos. 

Vamos a ajustar 3 modelos:
* Un modelo lineal (polinomio de grado 1).
* Un modelo con un polinomio de grado 3.
* Un modelo con un polinomio de grado 10.

Representamos los ajustes y mostramos el error. 


```python
# Vamos a crear la función para dibujar estos modelos
def plot_linear_regression_sklearn(reg, xlim, ax=None, annotate=False, poly=None, scaler=None): # mean_std_x es una tupla
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))
    x = np.linspace(*xlim).reshape(-1, 1)
    x_raw = x.copy()
    if poly:
        x = poly.transform(x)
        
    if scaler:
        x = scaler.transform(x)
    
    y = reg.predict(x) # aquí está el cambio
    ax.plot(x_raw, y, linewidth=4)
    if annotate:
        ax.annotate(r'$h_\theta(x)={:.2f} + {:.2f}x$'.format(reg.intercept_, reg.coef_), (np.min(x_raw), np.max(y)), fontsize=16)
        ax.annotate(r'$\theta_0 = {:.4f}, \theta_1 = {:.4f}$'.format(reg.intercept_, reg.coef_), (np.min(x_raw), np.max(y)*0.95), fontsize=16);
    return ax 
```


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

grados = [1, 3, 10]  #declaramos los grados con los que vamos a probar

fig, ax = plt.subplots(1, 3, figsize=(8*3, 6), sharey = True)

for i, grado in enumerate(grados):
    poly = PolynomialFeatures(degree = grado, include_bias = False)
    poly.fit(x)
    Xpoly = poly.transform(x)
    
    scaler = StandardScaler()
    scaler.fit(Xpoly)
    Xpoly = scaler.transform(Xpoly)
    
    regr = LinearRegression()
    regr.fit(Xpoly, y)
    
    error = mean_squared_error(regr.predict(Xpoly), y)
    
    xlim = (np.min(x), np.max(x))
    
    plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[i])
    plot_linear_regression_sklearn(regr, xlim = xlim, ax = ax.ravel()[i], poly = poly, scaler = scaler)
    ax.ravel()[i].set_title('Grado del polinomio = {}, Error = {:.2f}'.format(grado, error), fontsize=20)
    ax.ravel()[i].set_ylim(150,480)
```


    
![png](output_13_0.png)
    


A medida que aumentamos el grado del polinomio, el error de entrenamiento decrece. De hecho, en la gráfica de la derecha vemos que estamos consiguiendo obtener un error cercano a 0, ya que estamos modelando los datos de entrenamiento casi perfectamente. Sin embargo, podemos intuir que el modelo de la derecha **no** va a obtener buenos resultados con a la hora de predecir datos nuevos. Tenemos un problema de **overfitting**.

Vamos a utilizar estos mismos modelos para comprobar que con datos nuevos no conseguimos el comportamiento deseado. 


```python
#Obtenemos datos nuevos
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

xtest, ytest, df_test = gen_house_prices_nonlinear(10, seed=123456789)

grados = [1, 3, 10]  #declaramos los grados con los que vamos a probar

fig, ax = plt.subplots(1, 3, figsize=(8*3, 6), sharey = True)

for i, grado in enumerate(grados):
    poly = PolynomialFeatures(degree = grado, include_bias = False)
    poly.fit(x)
    Xpoly = poly.transform(x)
    
    scaler = StandardScaler()
    scaler.fit(Xpoly)
    Xpoly = scaler.transform(Xpoly)
    
    regr = LinearRegression()
    regr.fit(Xpoly, y)
    
    error = mean_squared_error(regr.predict(Xpoly), y)
    errorTest = mean_squared_error(regr.predict(scaler.transform(poly.transform(xtest))), ytest)
    
    xlim = (np.min(np.vstack((x,xtest))), np.max(np.vstack((x,xtest))))
    
    plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[i])
    ax.ravel()[i].plot(xtest, ytest, '.g', markersize=12)
    plot_linear_regression_sklearn(regr, xlim = xlim, ax = ax.ravel()[i], poly = poly, scaler = scaler)
    ax.ravel()[i].set_title('Grado = {}, Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, error, errorTest), fontsize=15)
    ax.ravel()[i].set_ylim(150,480)
```


    
![png](output_15_0.png)
    


El comportamiento del error con datos nuevos es el esperado: el modelo con características polinomiales de grado 3 obtiene menor error que el modelo lineal y ambos obtienen un error mucho menor al modelo entrenado con características polinomiales de grado 10. 

Esto se debe a que tenemos muchas características y pocos ejemplos, por lo que nuestro modelo se está ajustando perfectamente a las particularidades del conjunto de datos con el que estamos entrenando, perdiendo capacidad de generalización. 

### 6.1.2 Overfitting en un problema de clasificación

El overfitting se da de igual manera en problemas de clasificación. En esta sección, vamos a ilustrar este hecho volviendo al problema de clasificación de piezas metálicas: clasificar las piezas en las clases OK/NOK dependiendo del tamaño de la grieta detectada, para lo cual somos capaces de medir automáticamente su anchura y su profundidad.

Veámoslo con los siguientes datos.


```python
from sklearn.datasets import make_gaussian_quantiles

# Simulamos el problema
def gen_binary_classification(n, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Para poder replicar los resultados
        
    X1, y1 = make_gaussian_quantiles(mean =[0.7, 0.7], cov = 3, n_samples= n * 4, n_classes = 2, random_state = seed)
    X = X1[np.logical_and(X1[:,0]>0, X1[:,1]>0),:]
    y = y1[np.logical_and(X1[:,0]>0, X1[:,1]>0)] 
    X[y==1] = 0.8*X[y==1]
    

    df = pd.DataFrame(np.hstack((X, y.reshape(-1,1))), columns=[r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', 'Clase'])
    
    return X, y, df
    
X, y, df = gen_binary_classification(100, seed=12)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anchura ($\mu m$)</th>
      <th>Profundidad ($\mu m$)</th>
      <th>Clase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.947710</td>
      <td>2.477665</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.670111</td>
      <td>2.585644</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.221318</td>
      <td>1.106205</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.264972</td>
      <td>0.488562</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.513959</td>
      <td>0.416311</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Representamos los datos gráficamente:


```python
from matplotlib.colors import ListedColormap

# Creamos los mapas de colores a utilizar
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

def plot_data_clas(X, y, labelx, labely, xlim=None, ylim=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=cmap_bold)
    ax.legend(handles=scatter.legend_elements()[0], labels=map(lambda val: 'Clase ' + str(int(val)), np.unique(y)) )
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax

_ = plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)')
```


    
![png](output_21_0.png)
    


En la gráfica se ve que hay zonas donde los ejemplos pertenecen con claridad a una de las dos clases, pero la frontera entre ambas no está bien definida. 

Como antes, entrenamos tres modelos para mostrar el overfitting en problemas de clasificación. Los modelos corresponden a:
* Un modelo lineal (polinomio de grado 1).
* Un modelo con un polinomio de grado 2.
* Un modelo con un polinomio de grado 6.

Representamos los ajustes y mostramos el error. 


```python
def limit_helper(X):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    return (x_min, x_max), (y_min, y_max)

def plot_logistic_regression_sklearn(clasif, x1_lim, x2_lim, ax=None, scaler=None, poly=None):
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))

    # Usamos meshgrid para generar una malla de puntos con todas 
    # las combinaciones entre los vectores que recibe como parámetro
    xx, yy = np.meshgrid(np.linspace(*x1_lim, 200),
                         np.linspace(*x2_lim, 200))
    
    
    X = np.hstack((xx.reshape(-1, 1),
                     yy.reshape(-1, 1)))
    
    if poly:
        X = poly.transform(X)
    
    if scaler:
        X = scaler.transform(X)
    
    # Clasificamos los puntos con nuestro modelo
    Z = clasif.predict(X).reshape(xx.shape)

    # Pintamos las fronteras
    ax.pcolormesh(xx, yy, Z>0.5, cmap=cmap_light, shading='auto', alpha=0.35)
    ax.contour(xx, yy, Z, [0.5], linewidths=2, colors='k', linestyles='dashed'); # marca la frontera
    #ax.pcolormesh(xx, yy, Z) # así podemos visualizar la probabilidad en vez de la frontera
                                  
    # Establecemos los límites
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
                                  
    return ax 
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

grados = [1, 2, 6]  #declaramos los grados con los que vamos a probar

fig, ax = plt.subplots(1, 3, figsize=(8*3, 6), sharey = True)

for i, grado in enumerate(grados):
    poly = PolynomialFeatures(degree = grado, include_bias = False)
    poly.fit(X)
    Xpoly = poly.transform(X)
    
    scaler = StandardScaler()
    scaler.fit(Xpoly)
    Xpoly = scaler.transform(Xpoly)
    
    LogReg = LogisticRegression(penalty = 'none', solver = 'newton-cg')
    LogReg.fit(Xpoly, y)
    
    acc = accuracy_score(LogReg.predict(Xpoly), y)
    
    plot_logistic_regression_sklearn(LogReg, *limit_helper(X), poly = poly, scaler = scaler, ax = ax.ravel()[i])
    plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax.ravel()[i])
    ax.ravel()[i].set_title('Grado del polinomio = {}, Acierto = {:.2f}%'.format(grado, acc*100), fontsize=20)
```


    
![png](output_24_0.png)
    


Al igual que antes, a medida que aumentamos el grado del polinomio, decrece el error el error de entrenamiento (acertamos más ejemplos) y, con grados más altos, surge el problema del overfitting. 

Como antes, vamos a comprobar que el número de aciertos decrece para ejemplos con los que no hemos entrenado. 


```python
Xtest, ytest, _ = gen_binary_classification(25, seed=12345)

grados = [1, 2, 6]  #declaramos los grados con los que vamos a probar

fig, ax = plt.subplots(1, 3, figsize=(8*3, 6), sharey = True)

for i, grado in enumerate(grados):
    poly = PolynomialFeatures(degree = grado, include_bias = False)
    poly.fit(X)
    Xpoly = poly.transform(X)
    
    scaler = StandardScaler()
    scaler.fit(Xpoly)
    Xpoly = scaler.transform(Xpoly)
    
    LogReg = LogisticRegression(penalty = 'none', solver = 'newton-cg')
    LogReg.fit(Xpoly, y)
    
    acc = accuracy_score(LogReg.predict(Xpoly), y)
    accTest = accuracy_score(LogReg.predict(scaler.transform(poly.transform(Xtest))), ytest)
    
    plot_logistic_regression_sklearn(LogReg, *limit_helper(X), poly = poly, scaler = scaler, ax = ax.ravel()[i])
    plot_data_clas(Xtest,ytest, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax.ravel()[i])
    ax.ravel()[i].set_title('Grado = {}, Aciertos (Train/Test)= {:.2f}% / {:.2f}%'.format(grado, acc*100, accTest*100),
                            fontsize=15)
```


    
![png](output_26_0.png)
    


### 6.1.3 Posibles soluciones al overfitting

Hemos visto que, cuando tenemos muchas características y pocos ejemplos, es fácil que se produzca overfitting. ¿Cómo podemos solucionarlo? 

En el caso de que el elevado número de características sea debido al uso de características polinomiales, una solución es escoger un grado de polinomio menor. Más adelante veremos cómo elegir el grado del polinomio. 

Pero, si los atributos de los que disponemos no son el resultado de aplicar características polinomiales, ¿qué podemos hacer? 
* Una opción es reducir el número de características (manualmente o mediante algún método de selección de características). 

¿y si todas aportan algo?

En ese caso, usaremos una técnica llamada **regularización**.

La regularización consiste en reducir la magnitud de los parámetros $\theta_j$ para conseguir curvas, en regresión, y fronteras de decisión, en clasificación, más suaves. 

Reduciendo la magnitud de los $\theta_j$:
* Reducimos la flexibilidad del modelo.
* Reducimos las probabilidades de sobre-aprendizaje.

#### Ejemplo:

Penalizamos $\theta_3$ y $\theta_4$ haciendo que sean muy pequeños. 

![imagen.png](imagen.png)

## 6.2. Regularización en Regresión Lineal  <a class="anchor" id="2"></a>

En este apartado vamos a ver cómo aplicar la regularización a la regresión lineal. En primer lugar, cargaremos las funciones necesarias para ajustar un modelo de regresión lineal vistas el primer día de este Módulo y, en segundo lugar, veremos qué modificaciones hacen falta para aplicar regularización. En concreto, veremos:
* Modificación de la función de coste.
* Solución directa con regularización.
* Descenso por gradiente con regularización.

Nos basaremos en el mismo ejemplo de antes: la predicción del valor de casas en función de sus metros cuadrados.


```python
x, y, df = gen_house_prices_nonlinear(10, seed=2)
plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)')
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$m^2$</th>
      <th>Precio (miles €)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>268.854582</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.0</td>
      <td>382.576042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>82.0</td>
      <td>299.251078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103.0</td>
      <td>347.101816</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.0</td>
      <td>369.906348</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_32_1.png)
    


Cargamos las funciones necesarias:

* La función con el modelo de regresión lineal, para realizar predicciones:

$$  h_\theta(x) = \theta_0 + \theta_1 x_1 + \ldots + \theta_n x_n $$



```python
# Hagamos que nuestro modelo trabaje con todas las variables
# thetas es un vector columna y X la matriz de datos (ejemplos x variables)
def linear_regression_model(thetas, X, ones=False):
    # Comprobamos si la X viene con la columna de unos
    if not ones:
        # la añadimos
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = X.dot(thetas)
    return y
```

* La función de coste, para medir el error de ajuste:

$$ J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$



```python
def compute_cost(X, y, thetas, ones=False):    
    # Obtener h (la salida para cada ejemplo) y J a partir de la salida (error)
    h = linear_regression_model(thetas, X, ones)
    J = 1.0/(2.0 * y.shape[0])*np.sum((h - y)**2)
    
    return J
```

* La función para aprender los parámetros $\theta$ mediante solución directa.

$$ \theta = (X^TX)^{-1}X^Ty $$



```python
def solucion_directa_linreg(X, y):
    # Asumimos que X viene sin la columna de unos para theta_0 por lo que el primer paso es añadirla
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    thetas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    return thetas
```

* La función para aprender los parámetros $\theta$ mediante descenso por gradiente.

  Recordemos que las derivadas parciales de la función de coste $J$ con respecto a cada $\theta_j$ es: 

$$ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)}  \text{ para  } j=0,\ldots,n $$

  y por tanto, la actualización del descenso por gradiente queda como

$$ \theta_j = \theta_j - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)}   \text{ para  } j=0,\ldots,n   $$


```python
def grad_descent_linreg(X, y, alpha=0.01, num_iters=1500):
    J_history = np.zeros(num_iters)
    # el + 1 viene de que en X no viene la columna de 1s
    thetas = np.zeros((X.shape[1] + 1, 1)).reshape(-1, 1) # podemos inicializar aleatoriamente o a ceros
    
    # Asumimos que X viene sin la columna de 1s, la añadimos
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    for iter in np.arange(num_iters):
        h = linear_regression_model(thetas, X, ones=True)
        thetas = thetas - alpha*(1.0/y.shape[0])*(X.T.dot(h-y))
        J_history[iter] = compute_cost(X, y, thetas, ones=True)
        
    return thetas, J_history
```

De esta forma, ya podemos ajustar un modelo de regresión lineal a nuestros datos.


```python
# Vamos a añadir la posibilidad de normalizar
def plot_linear_regression(thetas, xlim, ax=None, annotate=False, poly = None, scaler = None): 
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))
    x = np.linspace(*xlim).reshape(-1, 1)
    x_raw = x.copy()
    if poly is not None:
        x = poly.transform(x)
    if scaler is not None:
        x = scaler.transform(x)
    
    y = linear_regression_model(thetas, x) # aquí está el cambio
    ax.plot(x_raw, y, linewidth=4)
    if annotate:
        ax.annotate(r'$h_\theta(x)={:.2f} + {:.2f}x$'.format(thetas[0, 0], thetas[1, 0]), (np.min(x_raw), np.max(y)), fontsize=16)
        ax.annotate(r'$\theta_0 = {:.4f}, \theta_1 = {:.4f}$'.format(thetas[0, 0], thetas[1, 0]), (np.min(x_raw), np.max(y)*0.95), fontsize=16);
    return ax  
```


```python
thetas = solucion_directa_linreg(x, y)
ax = plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)')
_ = plot_linear_regression(thetas, (x.min(), x.max()), ax = ax)
```


    
![png](output_43_0.png)
    


Recordemos que para entrenar un modelo con características polinomiales, debemos añadir dichas características a nuestros datos.


```python
# Añadimos características polinomiales
poly = PolynomialFeatures(degree = 3, include_bias = False)
poly.fit(x)
X = poly.transform(x)

# Normalizamos
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

thetas = solucion_directa_linreg(X, y)

ax = plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)')
_ = plot_linear_regression(thetas, (x.min(), x.max()), ax = ax, poly = poly, scaler = scaler)
```


    
![png](output_45_0.png)
    


### 6.2.1 Modificación de la función de coste

Para llevar a cabo la regularización, vamos a modificar la función de coste añadiendo un término que penalice la magnitud de los parámetros $\theta$, para que al llevar a cabo la minimización nos centremos en buscar un buen ajuste y, a la vez, minimizar la magnitud de estos parámetros. 

Además, vamos a usar un hiper-parámetro $\lambda \geq 0$ que nos va a servir para controlar hasta qué punto nos queremos centrar en ajustar el modelo a los puntos y hasta qué punto queremos reducir la magnitud de los $\theta$. Llamaremos a este hiper-parámetro el factor de regularización $\lambda$. 

La nueva función de coste es:

$$ J(\theta) = \frac{1}{2m} \left( \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2  + \lambda \sum_{j=1}^n \theta_j^2 \right)$$

**Nota:** El parámetro $\theta_0$ no se regulariza, ya que no acompaña a ninguna variable, sino que es el término independiente.


```python
def compute_cost_regu(X, y, thetas, lambdaReg=0, ones=False):    
    # Obtener h (la salida para cada ejemplo) y J a partir de la salida (error)
    h = linear_regression_model(thetas, X, ones)
    J = 1.0/(2.0 * y.shape[0])*(np.sum((h - y)**2) + lambdaReg * np.sum(thetas[1:]**2)) 
    
    return J
```

Minimizando esta función de coste, podemos utilizar mayor grado en las características polinomiales y, después de regularizar, obtendremos curvas más *suaves* que las que corresponden al grado de polinomio utilizado. 

La elección del parámetro $\lambda$ es importante para conseguir un buen modelo. 
* Si elegimos $\lambda = 0$, no estamos regularizando.
* Si elegimos un $\lambda$ adecuado, conseguiremos no sobre-aprender el modelo.
* Si elegimos $\lambda$ demasiado grande, estaremos regularizando demasiado y el modelo no se ajustará a los datos.

En el caso de tomar un valor de $\lambda$ muy grande ($\lambda = 10^{10}$, por ejemplo), lo que conseguiremos es que, a la hora de minimizar la función de coste, reducir la magnitud de los parámetros $\theta_j$ (con $j \in \{1, \ldots, n\}$) tenga prioridad absoluta.

De esta forma conseguiremos lo siguiente:

$$\theta_1 = \theta_2 = \ldots = \theta_n =0$$

y, por lo tanto,

$$h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \ldots + \theta_n x^n = \theta_0.$$




```python
thetas[1:] = 0

ax = plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)')
_ = plot_linear_regression(thetas, (x.min(), x.max()), ax = ax, poly = poly, scaler = scaler)
```


    
![png](output_50_0.png)
    


### 6.2.2 Solución directa

A pesar de haber modificado la función de coste, sigue existiendo una solución directa para encontrar los valores de los parámetros $\theta$ que minimizan la nueva función $J$. 

Si $X$ se refiere a la matriz de ejemplos con las variables de entrada (incluyendo una columna de unos correspondiente al $\theta_0$), la fórmula para obtener los parámetros $\theta = (\theta_0, \theta_1, \ldots, \theta_n)^T$ es la siguiente:

$$\theta = \left( X^T X + \lambda \begin{bmatrix}
0 & \cdots & 0 & 0 & 0\\
\vdots & 1 & 0 & \cdots & 0\\
0 & 0 & 1 & 0& 0\\
0 & \vdots & 0 & \ddots & 0\\
0 & 0 & 0 & 0 & 1\\
\end{bmatrix} \right)^{-1} X^T y$$

Además, regularizando tenemos un beneficio adicional:
* Si $\lambda >0$, **la matriz es invertible**.


```python
def solucion_directa_linreg_regu(X, y, lambdaReg=0):
    # Asumimos que X viene sin la columna de unos para theta_0 por lo que el primer paso es añadirla
    Xones = np.hstack((np.ones((X.shape[0], 1)), X))
    
    diagonal = np.ones((Xones.shape[1]))
    diagonal[0] = 0
    mat_diagonal = np.diag(diagonal) 
    
    thetas = np.linalg.pinv(Xones.T.dot(Xones) + lambdaReg * mat_diagonal).dot(Xones.T).dot(y)
    
    return thetas
```

Vamos a comparar los resultados de ajustar un polinomio de grado 10 sin regularizar y regularizando.


```python
fig, ax = plt.subplots(1, 2, figsize=(8*2, 6), sharey = True)

grado = 10

# Añadimos características polinomiales
poly = PolynomialFeatures(degree = grado, include_bias = False)
poly.fit(x)
X = poly.transform(x)

# Normalizamos
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


lambdas = [0, 0.1]

thetas = solucion_directa_linreg_regu(X, y, lambdas[0])

ypred = linear_regression_model(thetas, X)
error = mean_squared_error(ypred,y)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax[0])
plot_linear_regression(thetas, (x.min(), x.max()), ax = ax[0],
                       poly = poly, scaler = scaler)
ax[0].set_title('Grado = {}, $\lambda = $ {}, Error = {:.2f}'.format(grado, lambdas[0], error),
                            fontsize=20)

thetas = solucion_directa_linreg_regu(X, y, lambdas[1])
ypred = linear_regression_model(thetas, X)

error = mean_squared_error(ypred,y)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[1])
plot_linear_regression(thetas, (x.min(), x.max()), ax = ax.ravel()[1],
                       poly = poly, scaler = scaler)
ax[1].set_title('Grado = {}, $\lambda = $ {}, Error = {:.2f}'.format(grado, lambdas[1], error),
                            fontsize=20)
print()
```

    
    


    
![png](output_55_1.png)
    


Observamos que, con regularización, la curva de ajuste se suaviza y el modelo será capaz de generalizar mejor. 

Vamos a mostrar el error de los ejemplos de test creados anteriormente.


```python
fig, ax = plt.subplots(1, 2, figsize=(8*2, 6), sharey = True)

# Cargamos los ejemplos de test
xtest, ytest, _ = gen_house_prices_nonlinear(10, seed=123456789)

grado = 10

# Añadimos características polinomiales
poly = PolynomialFeatures(degree = grado, include_bias = False)
poly.fit(x)
X = poly.transform(x)
Xtest = poly.transform(xtest)

# Normalizamos
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Xtest = scaler.transform(Xtest)


lambdas = [0, 0.1]

thetas = solucion_directa_linreg_regu(X, y, lambdas[0])

ypred = linear_regression_model(thetas, X)
ypredTest = linear_regression_model(thetas, Xtest)
errorTrain = mean_squared_error(ypred,y)
errorTest = mean_squared_error(ypredTest,ytest)

xlim = (np.min(np.vstack((x,xtest))), np.max(np.vstack((x,xtest))))
ax[0].set_ylim(150,480)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax[0])
plot_linear_regression(thetas, xlim, ax = ax[0],
                       poly = poly, scaler = scaler)
ax[0].plot(xtest, ytest, '.g', markersize=12)
ax[0].set_title('Grado = {}, $\lambda = $ {}, Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, lambdas[0], 
                                                                                           errorTrain, errorTest),
                            fontsize=15)

thetas = solucion_directa_linreg_regu(X, y, lambdas[1])

ypred = linear_regression_model(thetas, X)
ypredTest = linear_regression_model(thetas, Xtest)
errorTrain = mean_squared_error(ypred,y)
errorTest = mean_squared_error(ypredTest,ytest)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[1])
plot_linear_regression(thetas, xlim, ax = ax.ravel()[1],
                       poly = poly, scaler = scaler)
ax[1].plot(xtest, ytest, '.g', markersize=12)
ax[1].set_title('Grado = {}, $\lambda = $ {}, Error = Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, lambdas[1], errorTrain, errorTest),
                            fontsize=15)
print()
```

    
    


    
![png](output_57_1.png)
    


Efectivamente, el modelo se comporta mejor ante datos nuevos. 

### 6.2.3 Descenso por gradiente

El algoritmo de descenso por gradiente también se ve modificado por los cambios introducidos en la función de coste. 

Los pasos del algoritmo no varían:
1. Asignamos a $\theta=\{\theta_0, \ldots, \theta_n \}$ valores aleatorios
2. Repetimos hasta convergencia (nº de iteraciones o $|J(\theta_{it}) - J(\theta_{it-1})| < umbral$ )

$$ \theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial\theta_j}, \text{ para } j=0,\ldots, n $$

Pero cambia la expresión de la derivada parcial $\frac{\partial J(\theta)}{\partial \theta_j}$.

Como $\theta_0$ no se regulariza, tenemos que distinguir entre $j=0$ y $j \in \{1, \ldots, n\}$ para realizar los cálculos:

\begin{align*}
 \frac{\partial J(\theta)}{\partial \theta_0} &= \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_0^{(i)}   \\
\frac{\partial J(\theta)}{\partial \theta_j} &= \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j, \text{      para } j=1,\ldots,n 
\end{align*}

Así, los pasos del algoritmo de **descenso por gradiente con regularización** quedan:

1. Asignamos a $\theta=\{\theta_0, \ldots, \theta_n \}$ valores aleatorios
2. Repetimos hasta convergencia (nº de iteraciones o $|J(\theta_{it}) - J(\theta_{it-1})| < umbral$ )

 \begin{align*}
 \theta_0 &= \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_0^{(i)}    \\
 \theta_j &= \theta_j - \alpha \left( \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j \right),    \text{ para  } j=1,\ldots,n 
 \end{align*}


```python
def grad_descent_linreg_regu(X, y, alpha=0.01, lambdaReg=0, num_iters=1500):
    J_history = np.zeros(num_iters)
    # el + 1 viene de que en X no viene la columna de 1s
    thetas = np.zeros((X.shape[1] + 1, 1)).reshape(-1, 1) # podemos inicializar aleatoriamente o a ceros
    
    # Asumimos que X viene sin la columna de 1s, la añadimos
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    m = y.shape[0]
    
    for iter in np.arange(num_iters):
        h = linear_regression_model(thetas, X, ones=True)
        thetas = thetas - alpha*(1.0/m)*(X.T.dot(h-y)) - alpha * lambdaReg/m * np.vstack((0,thetas[1:])) 
        J_history[iter] = compute_cost(X, y, thetas, ones=True)
    return thetas, J_history
```

Volvemos a mostrar el mismo ejemplo que en el apartado anterior, pero esta vez para características polinomiales de grado 7. 

La conclusión es la misma: la regularización hace que el modelo se ajuste menos a los datos de train y conseguimos un mejor error de test, una mejor generalización. 


```python
fig, ax = plt.subplots(1, 2, figsize=(8*2, 6), sharey = True)

# Cargamos los ejemplos de test
xtest, ytest, _ = gen_house_prices_nonlinear(10, seed=123456789)

grado = 7

# Añadimos características polinomiales
poly = PolynomialFeatures(degree = grado, include_bias = False)
poly.fit(x)
X = poly.transform(x)
Xtest = poly.transform(xtest)

# Normalizamos
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Xtest = scaler.transform(Xtest)


lambdas = [0, 0.03]

thetas, _ = grad_descent_linreg_regu(X, y, lambdaReg=lambdas[0], alpha = 0.01, num_iters = 100000)

ypred = linear_regression_model(thetas, X)
ypredTest = linear_regression_model(thetas, Xtest)
errorTrain = mean_squared_error(ypred,y)
errorTest = mean_squared_error(ypredTest,ytest)

xlim = (np.min(np.vstack((x,xtest))), np.max(np.vstack((x,xtest))))
ax[0].set_ylim(150,480)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax[0])
plot_linear_regression(thetas, xlim, ax = ax[0],
                       poly = poly, scaler = scaler)
ax[0].plot(xtest, ytest, '.g', markersize=12)
ax[0].set_title('Grado = {}, $\lambda = $ {}, Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, lambdas[0], 
                                                                                           errorTrain, errorTest),
                            fontsize=15)

thetas, _ = grad_descent_linreg_regu(X, y, lambdaReg=lambdas[1], alpha = 0.01, num_iters = 100000)

ypred = linear_regression_model(thetas, X)
ypredTest = linear_regression_model(thetas, Xtest)
errorTrain = mean_squared_error(ypred,y)
errorTest = mean_squared_error(ypredTest,ytest)

plot_data_reg(x, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[1])
plot_linear_regression(thetas, xlim, ax = ax.ravel()[1],
                       poly = poly, scaler = scaler)
ax[1].plot(xtest, ytest, '.g', markersize=12)
ax[1].set_title('Grado = {}, $\lambda = $ {}, Error = Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, lambdas[1], errorTrain, errorTest),
                            fontsize=15)
print()
```

    
    


    
![png](output_63_1.png)
    


## 6.3 Regularización en Regresión Logística <a class="anchor" id="3"></a>

En este apartado vamos a ver cómo aplicar la regularización a la regresión logística. De nuevo, cargaremos las funciones necesarias para ajustar un modelo de regresión logística y veremos qué modificaciones hacen falta para aplicar regularización. En concreto, veremos:
* Modificación de la función de coste.
* Descenso por gradiente con regularización.

Nos basaremos en el mismo ejemplo de antes: la clasificación de piezas metálicas.


```python
X, y, df = gen_binary_classification(100, seed=12)
plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)')
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anchura ($\mu m$)</th>
      <th>Profundidad ($\mu m$)</th>
      <th>Clase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.947710</td>
      <td>2.477665</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.670111</td>
      <td>2.585644</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.221318</td>
      <td>1.106205</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.264972</td>
      <td>0.488562</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.513959</td>
      <td>0.416311</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_66_1.png)
    


Cargamos las funciones necesarias:

* La función sigmoide:

$$ g(z) = \frac{1}{1 + e^{-z}} $$

* La función con el modelo de regresión lineal, para realizar predicciones:

$$  h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \ldots + \theta_n x_n) $$



```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_model(thetas, X, ones=False):
    # Comprobamos si la X viene con la columna de unos
    if not ones:
        # la añadimos
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = sigmoid(X.dot(thetas))
    return y
```

* La función de coste *binary cross-entropy*, para medir el error de ajuste:

$$ J(\theta) = -\frac{1}{m}\sum_{i=1}^m y^{(i)}\log(h_\theta(x^{(i)})) + (1 - y^{(i)})\log(1-h_\theta(x^{(i)}))  $$


```python
# Función de coste de la regresión logística
def binary_cross_entropy(X, y, thetas, ones=False):
    # Obtener h (la salida para cada ejemplo) y J a partir de la salida (error)
    h = logistic_regression_model(thetas, X, ones)
    # Evitamos problemas con log
    e = np.finfo(np.float32).eps
    h = np.clip(h, 0+e, 1-e)
    
    J = -1.0*(1.0 / X.shape[0])*(np.log(h).T.dot(y) + np.log(1.0 - h).T.dot(1.0 - y))
    
    return J[0]
```

* La función para aprender los parámetros $\theta$ mediante descenso por gradiente.

  Recordemos que las derivadas parciales de la función de coste $J$ con respecto a cada $\theta_j$ es: 

$$ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)},  \text{ para  } j=0,\ldots,n $$

  y por tanto, la actualización del descenso por gradiente queda como

$$ \theta_j = \theta_j - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)} ,  \text{ para  } j=0,\ldots,n   $$

**Nota:** Las expresiones de las derivadas parciales y la actualización de parámetros coinciden con la regresión lineal, pero recordemos que ahora nuestra hipótesis, $h_\theta$, es diferente.


```python
def grad_descent_logreg(X, y, alpha=0.1, num_iters=150):
    # el + 1 viene de que en X no viene la columna de 1s
    thetas = np.zeros((X.shape[1] + 1, 1)).reshape(-1, 1) # podemos inicializar aleatoriamente o a ceros
    
    # Asumimos que X viene sin la columna de 1s, la añadimos
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = y.reshape(-1,1)
    
    for iter in np.arange(num_iters):
        h = logistic_regression_model(thetas, X, ones=True)
        thetas = thetas - alpha*(1.0/y.shape[0])*(X.T.dot(h-y))
    return thetas
```

Así, ya podemos entrenar un modelo de regresión lineal para clasificar nuestros datos. 


```python
def plot_logistic_regression(thetas, x1_lim, x2_lim, ax=None, scaler=None, poly=None):
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))

    # Usamos meshgrid para generar una malla de puntos con todas 
    # las combinaciones entre los vectores que recibe como parámetro
    xx, yy = np.meshgrid(np.linspace(*x1_lim, 200),
                         np.linspace(*x2_lim, 200))
    
    
    X = np.hstack((xx.reshape(-1, 1),
                     yy.reshape(-1, 1)))
    
    if poly is not None:
        X = poly.transform(X)
    
    if scaler is not None:
        X = scaler.transform(X)
    
    # Clasificamos los puntos con nuestro modelo
    Z = logistic_regression_model(thetas, X).reshape(xx.shape)

    # Pintamos las fronteras
    ax.pcolormesh(xx, yy, Z>0.5, cmap=cmap_light, shading='auto', alpha=0.35)
    ax.contour(xx, yy, Z, [0.5], linewidths=2, colors='k', linestyles='dashed'); # marca la frontera
    #ax.pcolormesh(xx, yy, Z) # así podemos visualizar la probabilidad en vez de la frontera
                                  
    # Establecemos los límites
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
                                  
    return ax 
```


```python
thetas = grad_descent_logreg(X, y, alpha=0.1, num_iters=10000)
ax = plot_logistic_regression(thetas, *limit_helper(X))
_ = plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax)
```


    
![png](output_75_0.png)
    


Como antes, podemos añadir características polinomiales.


```python
X, y, df = gen_binary_classification(100, seed=12)

# Añadimos características polinomiales
poly = PolynomialFeatures(degree = 2, include_bias = False)
poly.fit(X)
Xpoly = poly.transform(X)

# Normalizamos
scaler = StandardScaler()
scaler.fit(Xpoly)
Xpoly = scaler.transform(Xpoly)

thetas = grad_descent_logreg(Xpoly, y, alpha=0.1, num_iters=10000)
ax = plot_logistic_regression(thetas, *limit_helper(X), poly = poly, scaler = scaler)
_ = plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax)
```


    
![png](output_77_0.png)
    


### 6.3.1 Modificación de la función de coste


Como en regresión lineal, vamos a modificar la función de coste añadiendo un término que penalice la magnitud de los parámetros $\theta$, controlando la regularización con un hiper-parámetro $\lambda \geq 0$. 

La nueva función de coste es:

$$ J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\big[y^{(i)}\, log\,( h_\theta\,(x^{(i)}))+(1-y^{(i)})\,log\,(1-h_\theta(x^{(i)}))\big] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2}$$

**Nota:** El parámetro $\theta_0$ no se regulariza, ya que no acompaña a ninguna variable, sino que es el término independiente.


```python
# Función de coste de la regresión logística
def binary_cross_entropy_regu(X, y, thetas, lambdaReg = 0, ones=False):
    # Obtener h (la salida para cada ejemplo) y J a partir de la salida (error)
    h = logistic_regression_model(thetas, X, ones)
    # Evitamos problemas con log
    e = np.finfo(np.float32).eps
    h = np.clip(h, 0+e, 1-e)
    
    m = X.shape[0]
    
    J = -1.0*(1.0 / m)*(np.log(h).T.dot(y) + np.log(1.0 - h).T.dot(1.0 - y)) + lambdaReg/2/m * np.sum(theta[1:]**2)
    
    return J[0]
```

Esta modificación tiene el mismo efecto que en el caso de la regresión lineal. 

### 6.3.2 Descenso por gradiente

En el caso de la regresión logística con regularización, la fórmula por la que se actualizan los parámetros en el descenso por gradiente coincide con la fórmula de la regresión lineal. Sin embargo, hay que tener en cuenta que la hipótesis $h_\theta$ es distinta. Los pasos del algoritmo de **descenso por gradiente con regularización** quedan:

1. Asignamos a $\theta=\{\theta_0, \ldots, \theta_n \}$ valores aleatorios
2. Repetimos hasta convergencia (nº de iteraciones o $|J(\theta_{it}) - J(\theta_{it-1})| < umbral$ )

 \begin{align*}
 \theta_0 &= \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_0^{(i)}    \\
 \theta_j &= \theta_j - \alpha \left( \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j \right),    \text{ para  } j=1,\ldots,n 
 \end{align*}


```python
def grad_descent_logreg_regu(X, y, alpha=0.1, lambdaReg = 0, num_iters=150):
    # el + 1 viene de que en X no viene la columna de 1s
    thetas = np.zeros((X.shape[1] + 1, 1)).reshape(-1, 1) # podemos inicializar aleatoriamente o a ceros
    
    # Asumimos que X viene sin la columna de 1s, la añadimos
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = y.reshape(-1,1)
    m = y.shape[0]
    
    for iter in np.arange(num_iters):
        h = logistic_regression_model(thetas, X, ones=True)
        thetas = thetas - alpha*((1.0/m)*(X.T.dot(h-y)) + lambdaReg/m * np.vstack((0,thetas[1:])) )
    return thetas
```

Veamos el resultado de aplicar regularización a un modelo de regresión logística que ha sido entrenado usando características polinomiales de grado 5. 


```python
fig, ax = plt.subplots(1, 2, figsize=(8*2, 6), sharey = True)

# Cargamos los ejemplos de test
X, y, _ = gen_binary_classification(100, seed=12)
Xtest, ytest, _ = gen_binary_classification(25, seed=12345)

grado = 5

# Añadimos características polinomiales
poly = PolynomialFeatures(degree = grado, include_bias = False)
poly.fit(X)
Xpoly = poly.transform(X)
Xpolytest = poly.transform(Xtest)

# Normalizamos
scaler = StandardScaler()
scaler.fit(Xpoly)
Xpoly = scaler.transform(Xpoly)
Xpolytest = scaler.transform(Xpolytest)


lambdas = [0, 3]

thetas = grad_descent_logreg_regu(Xpoly, y, lambdaReg=lambdas[0], alpha = 3.0, num_iters = 100000)

ypred = logistic_regression_model(thetas, Xpoly)
ypred = np.round(ypred).ravel()

ypredTest = logistic_regression_model(thetas, Xpolytest)
ypredTest = np.round(ypredTest).ravel()

accTrain = accuracy_score(ypred,y)
accTest = accuracy_score(ypredTest,ytest)

plot_logistic_regression(thetas, *limit_helper(X), poly = poly, scaler = scaler, ax = ax[0])
plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax[0])

ax[0].set_title('Grado = {}, $\lambda = $ {}, Aciertos (Train/Test) = {:.2f}% / {:.2f}%'.format(grado, lambdas[0], 100*accTrain, 100*accTest),
                fontsize=15)

thetas = grad_descent_logreg_regu(Xpoly, y, lambdaReg=lambdas[1], alpha = 3.0, num_iters = 100000)

ypred = logistic_regression_model(thetas, Xpoly)
ypred = np.round(ypred).ravel()

ypredTest = logistic_regression_model(thetas, Xpolytest)
ypredTest = np.round(ypredTest).ravel()

accTrain = accuracy_score(ypred,y)
accTest = accuracy_score(ypredTest,ytest)


plot_logistic_regression(thetas, *limit_helper(X), poly = poly, scaler = scaler, ax = ax[1])
plot_data_clas(X,y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax[1])

ax[1].set_title('Grado = {}, $\lambda = $ {}, Aciertos (Train/Test) = {:.2f}% / {:.2f}%'.format(grado, lambdas[1], 100*accTrain, 100*accTest),
                fontsize=15)
print()
```

    
    


    
![png](output_86_1.png)
    


## 6.4 Regularización en scikit-learn <a class="anchor" id="4"></a>

En esta sección vamos a ver que la posibilidad de regularizar está incluida en las clases de scikit-learn y vamos a aprender cómo utilizarla. 

### 6.4.1 Regresión lineal (SGDRegressor y Ridge)

Mientras que la clase `SGDRegressor`, que realiza la regresión por descenso por gradiente, permite utilizar regularización, la clase `LinearRegression` de scikit-learn, que utiliza el método directo, no permite regularización. Sin embargo, existe otra función para estimar los parámetros de la regresión lineal usando regularización: la clase `Ridge` (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

#### SGDRegressor

Los parámetros de esta función para utilizar regularización son:
* `penalty`: este parámetro hace referencia al *tipo* de regularización que vamos a utilizar. Puede tomar los valores {‘l2’, ‘l1’, ‘elasticnet’}. La que hemos visto en clase es la 'l2', que es la que toma por defecto.
* `alpha`: el factor de regularización (nuestro $\lambda$). Cuanto más grando sea este valor, más regularizaremos. Su valor por defecto es 0.0001. 

#### Ridge

Esta clase se utiliza para ajustar un modelo de regresión lineal en el que usamos regularización 'l2', la vista en este notebook. El factor de regularización se controla con siguiente parámetro:
* `alpha`: el factor de regularización (nuestro $\lambda$). Por defecto, toma el valor de 1.0. En esta clase se permite que este parámetro sea un array si queremos usar un factor de regularización distinto para cada variable. 

Vamos a ver un ejemplo de regresión lineal con regularización usando scikit-learn.


```python
# Podemos simplificar la función
def plot_linear_regression_sklearn(pipe, xlim, ax=None, annotate=False): # mean_std_x es una tupla
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))
    x = np.linspace(*xlim).reshape(-1, 1)
    
    y = pipe.predict(x) # aquí está el cambio y aquí se hace todo
    ax.plot(x, y, linewidth=4)
    if annotate:
        ax.annotate(r'$h_\theta(x)={:.2f} + {:.2f}x$'.format(pipe['reg'].intercept_, pipe['reg'].coef_), (np.min(x_raw), np.max(y)), fontsize=16)
        ax.annotate(r'$\theta_0 = {:.4f}, \theta_1 = {:.4f}$'.format(pipe['reg'].intercept_, pipe['reg'].coef_), (np.min(x_raw), np.max(y)*0.95), fontsize=16);
    return ax  
```


```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

lambdas = [0, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 10]

fig, ax = plt.subplots(3, 3, figsize=(8*3, 6*3), sharey = True)

# Cargamos los datos
X, y, _ = gen_house_prices_nonlinear(10, seed=2)
Xtest, ytest, _ = gen_house_prices_nonlinear(10, seed=123456789)

grado = 10

for i, valorLambda in enumerate(lambdas):

    pipe = Pipeline([('poly', PolynomialFeatures(grado, include_bias=False)), 
                     ('scaler', StandardScaler()), 
                     ('reg', Ridge(alpha = valorLambda))])

    pipe.fit(X, y.ravel())

    ypred = pipe.predict(X)
    ypredTest = pipe.predict(Xtest)
    errorTrain = mean_squared_error(ypred,y)
    errorTest = mean_squared_error(ypredTest,ytest)

    xlim = (np.min(np.vstack((X,Xtest))), np.max(np.vstack((X,Xtest))))
    ax.ravel()[0].set_ylim(150,480)

    plot_data_reg(X, y, 'Tamaño ($m^2$)', 'Precio (€)', ax = ax.ravel()[i])
    plot_linear_regression_sklearn(pipe, xlim, ax = ax.ravel()[i])
    ax.ravel()[i].plot(Xtest, ytest, '.g', markersize=12)
    ax.ravel()[i].set_title('Grado = {}, $\lambda = $ {}, Error (Train/Test) = {:.2f} / {:.2f}'.format(grado, valorLambda, 
                                                                                               errorTrain, errorTest), fontsize=15)
    
```


    
![png](output_92_0.png)
    


### 6.4.2 Regresión logística (SGDClassifier y LogisticRegression)

En el caso de la regresión logística nos fijamos en las clases `SGDClassifier` (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) y `LogisticRegression` (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) de scikit-learn, ya que ambas soportan la regularización.

#### SGDClassifier

Los parámetros de esta función para utilizar regularización son:
* `penalty`: este parámetro hace referencia al *tipo* de regularización que vamos a utilizar. Puede tomar los valores {‘l2’, ‘l1’, ‘elasticnet’}. La que hemos visto en clase es la 'l2', que es la que toma por defecto.
* `alpha`: el factor de regularización (nuestro $\lambda$). Cuanto más grando sea este valor, más regularizaremos. Su valor por defecto es 0.0001. 

#### LogisticRegression

Los parámetros de esta función para utilizar regularización son:
* `penalty`: este parámetro hace referencia al *tipo* de regularización que vamos a utilizar. Puede tomar los valores {‘l2’, ‘l1’, ‘elasticnet’, ‘none’}. Por defecto, toma la 'l2'. No todos los métodos de optimización, *solvers*, permiten todos los tipos de regularización. Por ejemplo, los métodos 'newton-cg', 'sag' y 'lbfgs' solo admiten l2. Si elegimos 'none', no se regulariza.
* `C`: el inverso del factor de regularización (lo interpretamos como $\frac{1}{\lambda}$). Por defecto, toma el valor de 1.0. Cuanto mayor sea el valor de *C* más nos ajustaremos a los datos y menos regularizaremos. 

Vamos a ver un ejemplo de regresión lineal con regularización usando scikit-learn.


```python
def plot_logistic_regression_sklearn(pipe, x1_lim, x2_lim, ax=None):
    if ax is None:
        fig, ax= plt.subplots(figsize=(8, 6))

    # Usamos meshgrid para generar una malla de puntos con todas 
    # las combinaciones entre los vectores que recibe como parámetro
    xx, yy = np.meshgrid(np.linspace(*x1_lim, 200),
                         np.linspace(*x2_lim, 200))
    
    
    X = np.hstack((xx.reshape(-1, 1),
                     yy.reshape(-1, 1)))
        
    # Clasificamos los puntos con nuestro modelo
    Z = pipe.predict(X).reshape(xx.shape)

    # Pintamos las fronteras
    ax.pcolormesh(xx, yy, Z>0.5, cmap=cmap_light, shading='auto', alpha=0.35)
    ax.contour(xx, yy, Z, [0.5], linewidths=2, colors='k', linestyles='dashed'); # marca la frontera
    #ax.pcolormesh(xx, yy, Z) # así podemos visualizar la probabilidad en vez de la frontera
                                  
    # Establecemos los límites
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
                                  
    return ax 
```


```python
lambdas = [0.0001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 10]

fig, ax = plt.subplots(3, 3, figsize=(8*3, 6*3), sharey = True)

## Cargamos los ejemplos de test
X, y, _ = gen_binary_classification(100, seed=12)
Xtest, ytest, _ = gen_binary_classification(25, seed=12345)

grado = 10

for i, valorLambda in enumerate(lambdas):

    pipe = Pipeline([('poly', PolynomialFeatures(grado, include_bias=False)), 
                     ('scaler', StandardScaler()), 
                     ('reg', LogisticRegression(solver='newton-cg', C = 1.0/valorLambda))])

    pipe.fit(X, y.ravel())

    ypred = pipe.predict(X)
    ypredTest = pipe.predict(Xtest)
    accTrain = accuracy_score(ypred,y)
    accTest = accuracy_score(ypredTest,ytest)

        
    plot_logistic_regression_sklearn(pipe, *limit_helper(X), ax = ax.ravel()[i])
    plot_data_clas(X, y, r'Anchura ($\mu m$)', 'Profundidad ($\mu m$)', ax = ax.ravel()[i])
    ax.ravel()[i].set_title('Grado = {}, $\lambda = $ {}, Aciertos (Train/Test) = {:.2f}% / {:.2f}%'.format(grado, valorLambda, 
                                                                                               100*accTrain, 100*accTest), fontsize=12)
    
```


    
![png](output_96_0.png)
    


## 6.5 Resumen y conclusiones <a class="anchor" id="5"></a>

Después de ver lo anterior sobre la regularización, esto es lo que debemos tener claro:
* La regularización sirve para evitar el overfitting 
* Es válida tanto en problemas de regresión como clasificación
* Consiste en una modificación de la función de coste
* Tenemos un hiper-parámetro ($\lambda$) con el que podemos controlar cuánto regularizamos nuestro modelo


```python

```
