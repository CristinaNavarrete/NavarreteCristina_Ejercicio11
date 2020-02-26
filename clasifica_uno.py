import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

%matplotlib inline

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8

data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

print(vectores)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

scores_train=[]
scores_test=[]

scores_train_ceros=[]
scores_test_ceros=[]


#proyecciones
proyecciones=x_train@vectores
proyecciones_test=x_test@vectores
y_train_transf=np.where(y_train!=1,0,y_train)
y_test_transf=np.where(y_test!=1,0,y_test)

for i in range(3,40):
    proyecciones_f=proyecciones[:,:i]
    clf = LinearDiscriminantAnalysis()
    clf.fit(proyecciones_f, y_train_transf)
    proyecciones_test_f=proyecciones_test[:,:i]
    y_predict_test=clf.predict(proyecciones_test_f)
    y_predict_train=clf.predict(proyecciones_f)
    scores_test.append(f1_score(y_test_transf, y_predict_test, ))
    scores_train.append(f1_score(y_train_transf,y_predict_train))
    scores_test_ceros.append(f1_score(y_test_transf, y_predict_test,pos_label=0))
    scores_train_ceros.append(f1_score(y_train_transf,y_predict_train,pos_label=0))
    
    
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("F1 scoresde los UNOS")
componentes=np.arange(3, 40, 1)
plt.scatter(componentes,scores_train,label='train')
plt.scatter(componentes,scores_test,label='test')
plt.legend()

plt.subplot(1,2,2)
plt.title("F1 scores de los NO UNOS")
componentes1=np.arange(3, 40, 1)
plt.scatter(componentes1,scores_train_ceros,label='train')
plt.scatter(componentes1,scores_test_ceros,label='test')
plt.legend()