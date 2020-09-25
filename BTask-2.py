from numpy import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing


df = pd.read_excel("C:\\Users\\NITIZEN\\Desktop\\Business_Ana\\BA new.xlsx")
#Transactions done on 22nd June
df = pd.read_excel("C:\\Users\\NITIZEN\\Desktop\\Business_Ana\\Bus_Task_2.xlsx")
df.replace(nan, int(0), inplace=True)

train = df.head(70)
#Given Values
operating_cost = 25
Aff_fee = 10

Cost_Of_Funds = (6.5/100) # charged by company to banks

Fix_Charge = 10

Cre_B_1 = 1000 # Getting increased by 5% every month.
Cre_B_2 = ((Cre_B_1 *5)/100 + Cre_B_1)
Cre_B_3 = ((Cre_B_1 *5)/100 + Cre_B_2)
Cre_B_4 = ((Cre_B_1 *5)/100 + Cre_B_3)
Cre_B_5 = ((Cre_B_1 *5)/100 + Cre_B_4)
Cre_B_6 = ((Cre_B_1 *5)/100 + Cre_B_5)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = "Cre_B_1", 'Cre_B_2', 'Cre_B_3', 'Cre_B_4',"Cre_B_5","Cre_B_6"
sizes = [1000, 1050, 1100, 1150,1200,1250]
explode = (0.8, 0.1, 0.2, 0.1,0.3,0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#interest applied due to borrower's default on monthly basis
int_a = (1000 *(1/12)*(15/100))
int_b = (Cre_B_2 * (1/12)*(15/100)) #interest(month-2) increase by 5% due to increase in credit balance.
int_c = (Cre_B_3 * (1/12)*(15/100))#interest(month-3) increase by 5% due to increase in credit balance.
int_d = (Cre_B_4 * (1/12)*(15/100))#interest(month-4) increase by 5% due to increase in credit balance.
int_e = (Cre_B_5 * (1/12)*(15/100))#interest(month-5) increase by 5% due to increase in credit balance.
int_f = (Cre_B_6 * (1/12)*(15/100))#interest(month-6) increase by 5% due to increase in credit balance.

labels = "int_a", 'int_b', 'int_c', 'int_d',"int_e","int_f"
sizes = [12.49, 13.125, 13.749, 14.374,15.0,15.624]
explode = (0.8, 0.5, 0.2, 0.1,0.2,0.3)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig2, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



Com_Int = 3/12 * 15 # compoud interest after three months

Loss_Rate = (3/100) # Risk on borrower's default
#If a person is unable to maintan average credit balance
Late_Fee = (3/100)

M_Fee1 = 20 # Grows by 5% everymonth
M_Fee2 = ((M_Fee1*5)/100 + M_Fee1)
M_Fee3 = ((M_Fee1*5)/100 + M_Fee2)
M_Fee4 = ((M_Fee1*5)/100 + M_Fee3)
M_Fee5 = ((M_Fee1*5)/100 + M_Fee4)
M_Fee6 = ((M_Fee1*5)/100 + M_Fee5)

Month_interest = (3.49/100)
Credit_Loss = df["Amount(Month1)"]*Loss_Rate
#Lets prepare our datasets.The days(used) are applied for all data of 6 months

plot = sns.boxplot(x ="Amount(Month1)",y = "Days(Used)",data = df,hue = "gender")
plt.grid()
plt.show()

D1 = df[df["Days(Used)"]<=0]
plt_1 = sns.boxenplot(x="Days(Used)",y="Amount(Month1)",hue="gender",data=D1)
plt.grid()
plt.show()

Profit_1 = (((D1["Amount(Month1)"]*Month_interest+ M_Fee1+ Aff_fee)-Cost_Of_Funds-(D1["Amount(Month1)"]*Loss_Rate)))
grph = plt.hist(Profit_1)
plt.xlabel("Profit'(Free_Period)")
plt.grid()
plt.show()

D2 = df[(df["Days(Used)"] >0) & (df["Days(Used)"]<=30)]
plt_2 = sns.violinplot(x="Days(Used)",y="Amount(Month1)",hue="gender",data=D2)
plt.grid()
plt.show()
late_fee = (D2["Amount(Month1)"]*int_b)/100
Charge = (D2["Amount(Month1)"]* Late_Fee)
Profit_2 = (((D2["Amount(Month1)"]*Month_interest+ M_Fee2+ late_fee)-Cost_Of_Funds- (D2["Amount(Month1)"]*Loss_Rate + Charge)))
grph_2 = plt.hist(Profit_2)
plt.xlabel("Profit(After Free Period)")
plt.grid()
plt.show()

D3 = df[(df["Days(Used)"] >30) & (df["Days(Used)"]<=45)]
late_fee_2 = (D3["Amount(Month1)"]*int_c)/100
Charge_1 = (D3["Amount(Month1)"]*Late_Fee)
Profit_3 = (((D3["Amount(Month1)"]*Month_interest+ M_Fee2+ late_fee_2)-Cost_Of_Funds-(D3["Amount(Month1)"]*Loss_Rate + Charge_1)))
grph_3 = plt.hist(Profit_3)
plt.xlabel("Profit(After 30 days)")
plt.grid()
plt.show()

D4 = df[(df["Days(Used)"] >45) & (df["Days(Used)"]<=60)]
late_fee_3 = (D4["Amount(Month1)"]*int_d)/100
Charge_2 = (D4["Amount(Month1)"]*Late_Fee)
Profit_4 = (((D4["Amount(Month1)"]*Month_interest+ M_Fee2+ late_fee_3)-Cost_Of_Funds-(D4["Amount(Month1)"]*Loss_Rate + Charge_2)))
grph_4 = plt.hist(Profit_4)
plt.xlabel("Profit(After 45 days)")
plt.grid()
plt.show()

D5 = df[(df["Days(Used)"] >60)]
late_fee_4 = (D5["Amount(Month1)"]*int_e)/100
Charge_3 = (D5["Amount(Month1)"]*Late_Fee)
Profit_5 = (((D5["Amount(Month1)"]*Month_interest+ M_Fee2+ late_fee_4)-Cost_Of_Funds-(D5["Amount(Month1)"]*Loss_Rate + Charge_3)))
grph_5 = plt.hist(Profit_5)
plt.xlabel("Profit(After 60 days)")
plt.grid()
plt.show()

#The k-means clustering

Train = df.head(100)
Train["Days(Used)"].map(lambda x: float(x))
Train.describe()
le = preprocessing.LabelEncoder()

X = Train.drop(columns = ["Days(Used)"]).values
Y = Train["Days(Used)"].values
le.fit(Train["gender"])
Train["enc_gender"] = le.transform(Train["gender"])
Test = df.loc[100:150]

train, validate, test = np.array_split(df.sample(frac = 1), [int(.6 * len(df)), int(.8 * len(df))])
# produces a 60%, 20%, 20% split for training, validation and test sets.


Train.info()
k_means = KMeans(n_clusters = 5)
est_kmeans = k_means.fit(Train[["Amount(Month1)", "Days(Used)"]].values)
gps = est_kmeans.predict(Test[["Amount(Month1)", "Days(Used)"]].values)
correct = 0
graph_4 = plt.scatter(x=Test["Amount(Month1)"].values, y=Test["Days(Used)"].values,c=gps, cmap="brg")
plt.grid()
plt.ylabel("Amount of transaction they made")
plt.xlabel("Days used by the person to pay")
plt.show()

#labels = "Cre_B_1", 'Cre_B_2', 'Cre_B_3', 'Cre_B_4',"Cre_B_5","Cre_B_6"
Test_a = df.loc[100:110]
x = Test_a["Days(Used)"].values
sizes = [x]
#explode = (0.8, 0.1, 0.2, 0.1,0.3,0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')

figa, ax1 = plt.subplots()
ax1.pie(sizes,autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()









