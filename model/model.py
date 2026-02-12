from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
df = pd.read_csv(r"C:\Users\princ\OneDrive\Pictures\Final Project of linear\Bike_resell_value_calc-main\Bike_resell_value_calc-main\LinearProjectModel\Used_Bikes.csv")
print(df.columns)
#step 1:  basic cleanup 
df.columns = [c.strip().lower() for c in df.columns]
# step 2 : set your target column name 
x = df.drop(columns=['price'])
y = df["price"]
# Columns Types 
num_cols = ['kms_driven','age','power'] 
cat_cols = ['bike_name','city','owner','brand']
# Pipeline is a way of doing work step by step in stages where multiple tasks are processeed at the same time but each one is in different stage 
# set pipeline 

# imputer : handles missing values (NAN)
# strategy ="mean" :  
# 1. Numberical data 
#  2. data normally distibuted 
# 3. outliers less 
# age marks, salary 
# median : numberical data data skewed  outliers present 
# income , salary , house price ,
# mode : mort_frequents : categorical data , numberic data with repeated values
# num_cols = ['kms_driven','age','power'] 
nums_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median"))
])
cat_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num',nums_pipeline,num_cols),
    ('cat',cat_pipeline,cat_cols)
])
# model 
model = LinearRegression()
pipe = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)
])
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2 , random_state=42
)
# train the model 
pipe.fit(x_train,y_train)

y_predict = pipe.predict(x_test)

print("R2 score is  : ",r2_score(y_test,y_predict))

import joblib
joblib.dump(pipe,"bike_price_model.pkl")