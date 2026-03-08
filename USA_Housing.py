from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('USA_Housing.csv')
df.info()
df.describe()
sns.pairplot(df)
sns.displot(df['Price'])
sns.heatmap(df.corr(numeric_only=True))
df.columns
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)
predictions=lm.predict(X_test)
print(predictions)
sns.displot(y_test-predictions)