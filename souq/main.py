import re
import pandas as pd
import matplotlib.pyplot as pl
from urllib import urlopen
from BeautifulSoup import BeautifulSoup as bs
import matplotlib

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
