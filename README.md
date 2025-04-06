### NAME: SURYA P <br>
### REG NO: 212224230280

# EX-5-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student


## AIM :

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## EQUIPMENTS REQUIRED :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## PROGRAM :

```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SURYA P
RegisterNumber:  212224230280

*/
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data (1).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## OUTPUT :

## DATA HEAD 

![Screenshot 2025-03-28 201701](https://github.com/user-attachments/assets/16e9446b-eacb-4b80-9a5a-29daa8a1240f)

## DATA1 HEAD

![Screenshot 2025-03-28 201706](https://github.com/user-attachments/assets/4b0b0e20-a767-49c2-9e3e-241519ab1b9b)

## ISNULL

![Screenshot 2025-03-28 201714](https://github.com/user-attachments/assets/919a1eba-617c-474f-8756-5bb32cc82816)

## DATA DUPLICATE

![Screenshot 2025-03-28 201717](https://github.com/user-attachments/assets/19148461-45e0-42d0-8ccc-ab0e9bebf210)
## DATA

![Screenshot 2025-03-28 201725](https://github.com/user-attachments/assets/869d51da-f754-4ac4-a248-45bf95960305)

## STATUS

![Screenshot 2025-03-28 201731](https://github.com/user-attachments/assets/0f3e7121-e83e-49b1-9649-55437f065154)

## Y_PRED

![Screenshot 2025-03-28 201738](https://github.com/user-attachments/assets/5c0e6d7d-fc5e-4561-9dc0-fec743b726fd)

## ACCURACY

![Screenshot 2025-03-28 201742](https://github.com/user-attachments/assets/40c4b15a-af74-420d-ba60-0c8bf163176d)

## CONFUSION MATRIX

![Screenshot 2025-03-28 201745](https://github.com/user-attachments/assets/37dfb6c9-56b3-4052-b597-4fa12254b47f)

## CLASSIFICATION

![Screenshot 2025-03-28 201808](https://github.com/user-attachments/assets/10db503c-5232-4325-a11c-b435992b9aa7)

## LR PREDICT

![Screenshot 2025-03-28 201833](https://github.com/user-attachments/assets/1e0eac7a-8e00-408d-81af-9e0a1b2856bd)


## RESULT :

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
