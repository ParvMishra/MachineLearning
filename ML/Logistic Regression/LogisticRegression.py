import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns

#sigmoid function
def sigmoid(z):

    a=1/(1+np.exp(-z))
    
    return a

#predicted y value for a given w,b ans x
def y_out(x_test,w_ini,b_ini):
    m,n=x_test.shape
    a=sigmoid(np.dot(w_ini,np.transpose(x_test))+b_ini)
    a=a.reshape(m,1)
    return a

#loss of a single training example
def loss(x_train,y_train,w_ini,b_ini):
    l=0
    l=-(y*np.log(y_out(x_train,w_ini,b_ini))+(1-y)*np.log(1-y_out(x_train,w_ini,b_ini)))
    
    return l

#the cost function
def cost(x,y,w,b):
    J=0
    m,n=x.shape
    J=(1/m)*np.sum(loss(x,y,w,b))
    
    return J

#derivative terms
def gradient(x_train,y_train,w_ini,b_ini):
    
    
    m,n=x_train.shape
    dw=np.zeros(n)
    dw=dw.reshape(len(dw),1)
    db=0.
    
    dw=(1/m)*(np.dot(np.transpose(x_train),y_out(x_train,w_ini,b_ini)-y_train))
    db=(1/m)*(np.sum(y_out(x_train,w_ini,b_ini)-y_train))
    
    dw=dw.reshape(1,len(dw))
    return dw,db


#gradient descent
def optimize(x_train,y_train,w_ini,b_ini,alpha,itirations):
    
    m,n=x.shape
    w=w_ini
    b=b_ini
    for i in range(itirations):
        
        dw,db=gradient(x_train,y_train,w,b)
        w=w-alpha*(dw)
        b=b-alpha*(db)
        
    return w,b

df=pd.read_csv("db.csv", encoding= 'unicode_escape')
# Define X and Y
x=df.iloc[:,0:7].values
y=df.iloc[:,8].values
y=y.reshape(len(y),1)


X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=0)
print("size of x train",X_train.shape)
print("size of y train",Y_train.shape)
print("size of x test",x_test.shape)
print("size of y test",y_test.shape)




m,n=x.shape
#initialization
w=np.zeros(n)
b=0.
#running the algoritm
w_out,b_out=optimize(x,y,w,b,0.000001,1000)


#parameters after running gradient descent
print(w_out,b_out)
y_pred = y_out(x_test,w_out, b_out)

pred=[]
for p in y_pred :
       if p >=0.45:
           pred.append(1)
       else:
           pred.append(0)
print(pred)

cm=confusion_matrix(y_test,pred)
print(cm)
accuracy_score(y_test,pred)

sns.regplot(x=y_pred, y=pred, data=df, logistic=True, ci=None)
