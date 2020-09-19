#Our first task is to find the profit earned by us in 1year.
import datetime as datetime
import random
import time
from numpy import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import sweetviz as sv
from sklearn.cluster import KMeans
from sklearn import preprocessing
data_1 = pd.read_excel("C:\\Users\\NITIZEN\\Desktop\\Business_Ana\\BA_1.xlsx")
#In order to fetch the imp parameters it is imp to plot and visualize them.
plt_1 = sns.barplot(x ="category",y = "Trans.Amt",data=data_1)
plt.grid()
plt.show()



df = pd.read_excel("C:\\Users\\NITIZEN\\Desktop\\Business_Ana\\BA new.xlsx")

sdf_2 = df['first'].str.cat(df['last'], sep =" ")
A = df["cc_num"].value_counts()
B = df["trans_date"].value_counts()

#Transactions done on 22nd June
cdf_1 = pd.read_excel("C:\\Users\\NITIZEN\\Desktop\\Business_Ana\\Busi_Task-3.xlsx")
cdf_1.replace(nan, int(0), inplace=True)

train = cdf_1.head(70)
#Given Values
operating_cost = 25
Aff_fee = 10
Cost_Of_Funds = (6.5/100)
Fix_Charge = 10
credit_balance = 1000
interest = (1000 *(31/365)*(15/100))
Loss_Rate = (3/100)
#Here we filtered out the values based upon duration took by the person
#People submit amount in free period

j1 = cdf_1[cdf_1["Days(Used)"]<=0]
print(j1.head())

#let's visualize the same:
plot = sns.boxenplot(x="Days(Used)",y="Trans_Amt",hue="gender",data=j1)
plt.grid()
plt.show()

#Now lets calculate the profit hat we will earn according to category of duration
Month_interest = (3.49/100)
Credit_Loss = j1["Total_Amt"]*Loss_Rate

#Profit earned in free period
Profit = ((j1["Total_Amt"]*Month_interest + operating_cost + Aff_fee+Fix_Charge)-(Cost_Of_Funds-(j1["Total_Amt"]*Loss_Rate)))
graph = plt.hist(Profit)
plt.grid()
plt.show()

#People submit amount after 30 days of free period (here free period = due date )
j2 = cdf_1[(cdf_1["Days(Used)"] >0) & (cdf_1["Days(Used)"]<=40)]
print(j2.head())
Credit_Loss_1 = j2["Total_Amt"]
#Profit Earned after 30 days
Late_Fee = interest * 30
Profit_2 = (((j2["Total_Amt"]*Month_interest + operating_cost + Aff_fee+Fix_Charge)+Late_Fee)-(Cost_Of_Funds-(j2["Total_Amt"]*Loss_Rate)))
graph_1 = plt.hist(Profit_2)
plt.grid()
plt.show()

plot_a = sns.barplot(x="Days(Used)",y="Trans_Amt",hue="gender",data=j2)
plt.grid()
plt.show()

#People submit amount after 45days of free period
j3 = cdf_1[(cdf_1["Days(Used)"] >40) & (cdf_1["Days(Used)"]<=55)]
print(j3.head())

#Profit Earned after 45 days
Late_Fee_2 = interest * 45
Profit_3 = (((j3["Total_Amt"]*Month_interest + operating_cost + Aff_fee+Fix_Charge)+Late_Fee_2)-(Cost_Of_Funds-(j3["Total_Amt"]*Loss_Rate) ) )
graph_2 = plt.hist(Profit_3)
plt.grid()
plt.show()

plot_b = sns.barplot(x="Days(Used)",y="Trans_Amt",hue="gender",data=j3)
plt.grid()
plt.show()

#People submit amount after 60 days of free period
j4 = cdf_1[(cdf_1["Days(Used)"] >55) & (cdf_1["Days(Used)"]<=70)]
print(j4.head())

#Profit after 60 days

Late_Fee_3 = interest * 60
Profit_4 = (((j4["Total_Amt"]*Month_interest + operating_cost + Aff_fee+Fix_Charge)+Late_Fee_3)-(Cost_Of_Funds-(j4["Total_Amt"]*Loss_Rate)))
graph_3 = plt.hist(Profit_4)
plt.grid()
plt.show()

plot_c = sns.barplot(x="Days(Used)",y="Trans_Amt",hue="gender",data=j4)
plt.grid()
plt.show()
#Lets visualize the credit looses for every date.

plot_1 = sns.barplot(x="Credit_Loss(22)",y="Trans_amt",hue="gender",data=cdf_1)
plt.grid()
plt.show()
  # |-> So there exist a linear relation between credit loss and transacton amount
#cdf_1["Days(Used)"] = cdf_1["Days(Used)"].map(lambda x: int(x))

#Time to prepare a k-means cluster based on same data which defines the
# people paid money according to duration

Train = cdf_1.head(100)
Train["Days(Used)"].map(lambda x: float(x))
Train.describe()
le = preprocessing.LabelEncoder()

X = Train.drop(columns = ["Days(Used)"]).values
Y = Train["Days(Used)"].values
le.fit(Train["gender"])
Train["enc_gender"] = le.transform(Train["gender"])
Test = cdf_1.loc[100:150]

train, validate, test = np.array_split(df.sample(frac = 1), [int(.6 * len(df)), int(.8 * len(df))])
# produces a 60%, 20%, 20% split for training, validation and test sets.


Train.info()
k_means = KMeans(n_clusters = 4)
est_kmeans = k_means.fit(Train[["Total_Amt", "Days(Used)"]].values)
gps = est_kmeans.predict(Test[["Total_Amt", "Days(Used)"]].values)
correct = 0
graph_4 = plt.scatter(x=Test["Total_Amt"].values, y=Test["Days(Used)"].values,c=gps, cmap="brg")
plt.grid()
plt.ylabel("Days used by the person to pay")
plt.xlabel("Amount of transaction they made")
plt.show()










