import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

df = pd.read_csv("./data_brute/GOOGL.P1D.csv")
df = df.rename(columns={'Unnamed: 0': 'Date'})
#df = df[['Date','Price']] 
df_date = df['Date']
df_price = df['price']

print(df_price)
df_price = np.log(df_price) #on prend le log pour stabiliser la variance, ne pas oublier de passer par l'exponentielle pour traduire
msk = (df.index < len(df)-30) #df.index représente l'indice du tableau qui va varier 
df_train = df_price[msk].copy() # on découpe afin d'effectuer nos tests, data pr l'entraînement 
df_test = df_price[~msk].copy() # data pour les tests 

#Déterminer si la série est stationnaire 

#Méthode 1 : Regarder la courbe 

"""
df_train.plot()
df_test.plot()
plt.show()
"""

#Méthode 2 : Utiliser des analyseurs graphiques

"""
acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)
plt.show()
"""

#Méthode 3 : utiliser une méthode numérique (ça va nous être utile pour l'implémentation car l'ordinateur ne "voit pas des courbes")
# ADF test : ça va nous retourner un p, si p< 5% c'est stationnaire, sinon non-stationnaire.
 

adf_test = adfuller(df_train)
# print(adf_test[1]) c'est la valeur de p qui nous intéresse 

# On différencie pour rendre la série stationnaire : 
df_train_diff = df_train.diff().dropna()

adf_test = adfuller(df_train_diff)
print(adf_test)


def stationary(df): # cette fonction prend en entrée un data frame, et renvoie le data frame stationnaire, et le nb de différenciations
    d=0 # nb de différenciations
    p = adfuller(df)[1]
    while p>0.05:
        df = df.diff().dropna()
        p = adfuller(df)[1]
        d += 1
    return df,d

df_train_diff,d = stationary(df_train)
#print(d) # on a la valeur du paramètre d 

# Choix de p et q : cf cours 
# on se place dans un modèle ARIMA(1,1,0) p=1 car pic au lag 1 dans PACF et pas après (on est avec AMAZON)
# On a essayé avec autoarima et il donne (0,1,0)
# Faire un code pour trouver p et q 

# Implémentation d'ARIMA
model = ARIMA(df_train,order=(3,3,3))
model_fit = model.fit()
#print(model_fit.summary())

# Etudier les résidus 
"""
residuals = model_fit.resid[1:]
fig,ax = plt.subplots(1,2)
residuals.plot(title = 'Residuals', ax = ax[0])
residuals.plot(title = 'Density',kind ='kde', ax = ax[1])
plt.show()

acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)
plt.show()
"""

# Prédire le modèle 
forecast_test = model_fit.forecast(len(df_test))

# Possibilité d'avoir les parmètres du modèle ici :
auto_arima = pm.auto_arima(df_train,stepwise = False,seasonal = False)
#print(auto_arima)
"""
# Tracé 
forecast_test = np.array(forecast_test)
df_price = np.array(df_price)
df_train = np.array(df_train)
df_predict = np.concatenate((df_train, forecast_test), axis=0)
x = np.array([i for i in range(len(df_train)+len(df_test))])

# On repasse à l'exponentielle car on avait pris le log avant 
df_predict = np.exp(df_predict)
df_price = np.exp(df_price)

plt.plot(x,df_predict)
plt.plot(x,df_price)
plt.show()

"""




