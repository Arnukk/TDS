import re
import pandas as pd
import matplotlib.pyplot as pl
from urllib import urlopen
from BeautifulSoup import BeautifulSoup as bs
import matplotlib.pyplot as pl
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model.logistic import LogisticRegression as LR


x = np.dot(np.random.randn(1000, 2), [[1, 0], [0, 2]])
x = np.dot(x, [[np.cos(np.pi/3), -np.sin(np.pi/3)], [np.sin(np.pi/3), np.cos(np.pi/3)]])

cov = np.dot(x.T, x)/999

v = np.array([[1], [0]])
for count in range(20):
    v = np.dot(cov, v)
    v = v/np.sqrt((v**2).sum())

print v[1]/v[0]
print np.tan(np.pi/6)
pl.plot(x[:, 0], x[:, 1], '.')
pl.show()

pca = PCA(n_components=1)
pca.fit(x)

print pca.explained_variance_
print np.cov(np.dot(x, v).T)

x = load_iris()['data']
y = load_iris()['target']

lr = LR()
lr.fit(x, y)

lr.score(x, y) #accuracy

tx = np.dot(x, lr.coef_.T)

pl.plot(tx)
pl.show()

pca2 = PCA(n_components=2)
pca2.fit(x)
px = pca2.transform(x)

pl.plot(px[0:50, 0], px[0:50, 1], 'r.')
pl.plot(px[50:100, 0], px[50:100, 1], 'gx')
pl.plot(px[100:150, 0], px[100:150, 1], 'ko')
pl.show()

pw = pca2.transform(lr.coef_)
for c in range(3):
    pl.plot([-pw[c,0], pw[c,0]], [-pw[c,1], pw[c,1]])
exit()

souqtext=urlopen("http://uae.souq.com/ae-en/").read()
souq=bs(souqtext)
items=souq.findAll('div',{'class':re.compile('placard')})

nreg=re.compile('\d+(?:,\d+)*')
rcom=lambda tmp:re.sub(',','',tmp)
def placard(intag):
    if intag.find('div','stars'):
        star = str(intag.find('div','stars').find('div','colored'))[str(intag.find('div','stars').find('div','colored')).find('width'):].rsplit(';', 1)[0].rsplit(':', 1)[1]
    else:
        star = 'No'
    return [intag.h6.text,int(rcom(nreg.search(intag.h5.find('span','is block').text).group())), star]

souqdf=pd.DataFrame([placard(tmp) for tmp in items],columns=['item','price','stars'])
souqdf[souqdf.price==souqdf.price.max()]
pl.figure()
souqdf.price.plot(kind='kde')
pl.show()






exit()
