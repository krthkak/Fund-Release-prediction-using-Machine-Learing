import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#data preprocesssing
#importing dataset
data = pd.read_csv("microlending_data.csv")
col = list(data.columns)

#importing the dataset with limited columns
dataset = pd.read_csv("microlending_data.csv",usecols = col)

#Joining the dummy varibles created for [gender,orrower_genders,activity,country,repayment_interval]
dataset = dataset.join(pd.get_dummies(dataset.borrower_genders))
dataset = dataset.join(pd.get_dummies(dataset.activity))
dataset = dataset.join(pd.get_dummies(dataset.country))
dataset = dataset.join(pd.get_dummies(dataset.repayment_interval))


#Dropping columns [gender,orrower_genders,activity,country,repayment_interval]
dataset = dataset.drop("borrower_genders",axis = 1)
dataset.drop("original_language",axis = 1,inplace = True)
dataset.drop("activity",axis = 1,inplace = True)
dataset.drop("country",axis = 1,inplace = True)
dataset.drop("sector",axis = 1,inplace = True)
dataset.drop("repayment_interval",axis = 1,inplace = True)
dataset.drop("status",axis = 1,inplace = True)

#droping missing value rows
dataset.dropna()

#Encoding company policy and status coulmn
ma = {"shared":1,"not shared":0}
status = {"funded":1,"not_funded":0}
years = {"1 Year":12 , "2 Years":24}
dataset.replace({"currency_policy":ma,"status":status},inplace = True)
dataset.replace({"term_in_months":years},inplace = True)
dataset.term_in_months = pd.to_numeric(dataset.term_in_months)

#Creating dependent varible
y = dataset.status.values
y = y.reshape(-1,1)

#creating independent variable
x = dataset.iloc[:,:].values

#Splitting dataset
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.3,random_state = 0)

#Model Logistics regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

#predicting values
y_pred = lr.predict(x_test)
y_pred_prob = lr.predict_proba(x_test)

#analysing accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

lr_as = accuracy_score(y_test , y_pred)
lr_cm = confusion_matrix(y_test,y_pred)
lr_cr = classification_report(y_test,y_pred)



tre_accuracy_score = {}
tre_confusion_matrix = {}
diff_tres = list(np.arange(0,1.1,0.1))

#precision curve
from sklearn.metrics import precision_recall_curve,auc
pre,rec,thr = precision_recall_curve(y_test,y_pred_prob[:,1])

rp_auc = auc(rec,pre)


#plotting 
plt.title("Precision-Recall vs Threshold")
plt.plot(thr,pre[:-1],"b--",label = "Precision")
plt.plot(thr,rec[:-1],"r--",label = "Recall")
plt.xlabel("Threshold")
plt.ylabel("Precision, Recall")
plt.legend(loc = "lower left")
plt.ylim([0,1])
plt.show()

#from y_pred
pre_pred,rec_pred,thr_pred = precision_recall_curve(y_test,y_pred)
pred_auc = auc(rec_pred,pre_pred)

plt.title("Precision vs Recall")
plt.plot(pre_pred,rec_pred,color = "red")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend()
plt.show()

