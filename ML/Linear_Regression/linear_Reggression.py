import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Import Dataset
df=pd.read_csv("airfoil_self_noise.csv", encoding= 'unicode_escape')

#df.head()

# Define X and Y
x=df.drop(df.columns[[5]], axis=1).values
y=df[df.columns[[5]]].values

# Spliting the data into train and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

# Train the dataset
ml=LinearRegression()
ml.fit(x_train,y_train)

# Predict the dataset
y_pre=ml.predict(x_test)
#print(y_pre)

# Metrices
r2_score(y_test,y_pre)

# Plot the graph
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pre)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')