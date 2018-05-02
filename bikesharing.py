import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataframe=pd.read_csv("D:\\bikesharing\\data\\train.csv")
sns.set()


dataframe["hour"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[1].split(":")[0]))
dataframe["month"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[0].split("-")[1]))
dataframe["season"]=dataframe.season.map({1:"spring",2:"summer",3:"fall",4:"winter"})
dataframe["workingday"]=dataframe.workingday.map({0:"no",1:"yes"})
dataframe["weather"]=dataframe.weather.map({1:"Clear",2:"Mist + Cloudy",3:"Light Snow",4:"Heavy Rain"})

#Count distribution

quartiles = pd.cut(dataframe["count"],10)
def get_stats(group):
    return {'counts': group.sum()}

grouped = dataframe["count"].groupby(quartiles)
count_distribution_amount = grouped.apply(get_stats).unstack()
df=pd.DataFrame(count_distribution_amount)



df.plot(kind='bar',figsize=(9, 7),rot=45)
plt.title("count distribution")
plt.show()
#Count distribution approximate to normal distribution


#Removal of outliers 

upper_bound=dataframe['count'].mean()+3*dataframe['count'].std()
lower_bound=dataframe['count'].mean()-3*dataframe['count'].std()
dataframe=dataframe[dataframe["count"]<upper_bound]
dataframe=dataframe[dataframe["count"]>lower_bound]

#weather statistics

Clear_casual=dataframe[dataframe["weather"]=="Clear"]["casual"].sum()
Mist_casual=dataframe[dataframe["weather"]=="Mist + Cloudy"]["casual"].sum()
Snow_casual=dataframe[dataframe["weather"]=="Light Snow"]["casual"].sum()
Rain_casual=dataframe[dataframe["weather"]=="Heavy Rain"]["casual"].sum()

Clear_registered=dataframe[dataframe["weather"]=="Clear"]["registered"].sum()
Mist_registered=dataframe[dataframe["weather"]=="Mist + Cloudy"]["registered"].sum()
Snow_registered=dataframe[dataframe["weather"]=="Light Snow"]["registered"].sum()
Rain_registered=dataframe[dataframe["weather"]=="Heavy Rain"]["registered"].sum()

index=["Clear","Mist","Snow","Rain"]
values1=[Clear_registered,Mist_registered,Snow_registered,Rain_registered]
values2=[Clear_casual,Mist_casual,Snow_casual,Rain_casual]
plt.bar(index,values1,color='blue')
plt.bar(index,values2,color='red',bottom=values1)
plt.title("Weather statistics bar chart ")
plt.legend(["number of registered user ","number of non-registered user"])
plt.show()
#Most of the data is on clear days,there is little data on rainy days


# hours statistics with weather

clear_data=dataframe[dataframe["weather"]=="Clear"]
mist_data=dataframe[dataframe["weather"]=="Mist + Cloudy"]
snow_data=dataframe[dataframe["weather"]=="Light Snow"]
rain_data=dataframe[dataframe["weather"]=="Heavy Rain"]

y1=clear_data.groupby('hour')['count'].mean()
y2=mist_data.groupby('hour')['count'].mean()
y3=snow_data.groupby('hour')['count'].mean()
#y4=rain_data.groupby('hour')['count'].mean()  Only one line
y4_data=[0]*24
y4_data[17]=164
x1=range(24)
y4=pd.DataFrame(y4_data,range(24),)
plt.plot(x1,y1,'ro-',x1,y2,'bo-',x1,y3,'go-',x1,y4,'yo-')
plt.title("Hour mean statistics with weather ")
plt.legend(["number of user in clear","number of user in mist","number of user in snow","number of user in rain"])
plt.xlabel("Hours")
plt.ylabel("Number of users")
plt.show()
#The morning peak is at eight in the morning and the evening peak is at seventeen in the evening


#month statistics
z1=clear_data.groupby('month')['count'].mean()
z2=mist_data.groupby('month')['count'].mean()
z3=snow_data.groupby('month')['count'].mean()
#z4=rain_data.groupby('month')['count'].mean() Only one line
z4_data=[0]*12
z4_data[0]=164
z4=pd.DataFrame(z4_data,range(12),)
x2=range(1,13)

plt.plot(x2,z1,'ro-',x2,z2,'bo-',x2,z3,'go-',x2,z4,'yo-')
plt.title("Month mean statistics with weather ")
plt.legend(["user in clear","user in mist","user in snow","user in rain"])
plt.xlabel("Month")
plt.ylabel("Number of users")
plt.xticks(range(1,13))
plt.show()
#January has the fewest users

#season statistics
m1=clear_data.groupby('season')['count'].sum()
m2=mist_data.groupby('season')['count'].sum()
m3=snow_data.groupby('season')['count'].sum()
m4=rain_data.groupby('season')['count'].sum()
x3=range(4)
bw=0.3
index=np.arange(4)
plt.bar(index,m1,bw)
plt.bar(index+bw,m2,bw)
plt.bar(index+2*bw,m3,bw)
plt.bar(index+3*bw,m4,bw)
plt.title("Season statistics with weather ")
plt.xticks(index+1.5*bw,['fall','spring','summer','winter'])
plt.legend(["user in clear","user in mist","user in snow","user in rain"])
plt.show()
#Summer and fall has more users and there is less users in spring and winter



#Correlation matrix
corr = dataframe.corr()
corr=corr.round(2)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap='coolwarm')
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.title("Correlation matrix")
plt.show()