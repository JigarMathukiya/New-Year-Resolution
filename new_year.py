#Import other necessary libraries like pandas and sklearn
import pandas as pd
from sklearn.metrics import accuracy_score

#Import Library
#from sklearn.linear_model import LogisticRegression
#from sklearn import tree
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier


#Load Train and Test datasets
x_train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')

#Replace data string to numeric
x_train['credit_score'].replace(['A+','A','B+','B','C+','C','I','P','U'],[0,1,2,3,4,5,6,7,8],inplace=True)
x_train['location_employee_code'].replace(['A','B','C','D','E','F','G','H','I','J','K'],[0,1,2,3,4,5,6,7,8,9,10],inplace=True)

x_test['credit_score'].replace(['A+','A','B+','B','C+','C','I','P','U'],[0,1,2,3,4,5,6,7,8],inplace=True)
x_test['location_employee_code'].replace(['A','B','C','D','E','F','G','H','I','J','K'],[0,1,2,3,4,5,6,7,8,9,10],inplace=True)

x_train.describe()
#print(x_train)

x = x_train.drop(['total_sales'], axis=1)
y = x_train.total_sales
#y = x_train.pop('total_sales')

#Identify feature and response variable(s) and values must be numeric
numeric_train=list(x.dtypes[x.dtypes != 'object'].index)
x.tail()

#Import Library
#model = LogisticRegression()
#model = tree.DecisionTreeClassifier(criterion='gini')
#model = GaussianNB()
model = KNeighborsClassifier(n_neighbors=5)
#model= RandomForestClassifier()


# Train the model using the training sets and check score
model.fit(x[numeric_train], y)
model.score(x[numeric_train], y)

# Train Accuracy
print('Train Accuracy : ', accuracy_score(y, model.predict(x[numeric_train]))*100)

#Identify feature and response variable(s) and values must be numeric
numeric_test=list(x_test.dtypes[x_test.dtypes != 'object'].index)

#Predict data
predicted= model.predict(x_test[numeric_test])

#Create submission file
submission=pd.DataFrame({"outlet_no": x_test["outlet_no"], "total_sales_actual":predicted})
submission.to_csv('submission.csv',index=False)
print(submission.head())
