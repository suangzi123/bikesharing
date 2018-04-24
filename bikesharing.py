import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataframe=pd.read_csv("D:\\bikesharing\\data\\train.csv")



dataframe["hour"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[1].split(":")[0]))
dataframe["month"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[0].split("-")[1]))
dataframe["season"]=dataframe.season.map({1:"spring",2:"summer",3:"fall",4:"winter"})
dataframe["workingday"]=dataframe.workingday.map({0:"no",1:"yes"})
dataframe["weather"]=dataframe.weather.map({1:"Clear",2:"Mist + Cloudy",3:"Light Snow",4:"Heavy Rain"})

#Clear_counts=dataframe[dataframe["weather"]=="Clear"]
#Mist_counts=dataframe[dataframe["weather"]=="Mist + Cloudy"]
#Snow_counts=dataframe[dataframe["weather"]=="Light Snow"]
#Rain_counts=dataframe[dataframe["weather"]=="Heavy Rain"]


Clear_casual=dataframe[dataframe["weather"]=="Clear"]["casual"].sum()
Mist_casual=dataframe[dataframe["weather"]=="Mist + Cloudy"]["casual"].sum()
Snow_casual=dataframe[dataframe["weather"]=="Light Snow"]["casual"].sum()
Rain_casual=dataframe[dataframe["weather"]=="Heavy Rain"]["casual"].sum()

Clear_registered=dataframe[dataframe["weather"]=="Clear"]["registered"].sum()
Mist_registered=dataframe[dataframe["weather"]=="Mist + Cloudy"]["registered"].sum()
Snow_registered=dataframe[dataframe["weather"]=="Light Snow"]["registered"].sum()
Rain_registered=dataframe[dataframe["weather"]=="Heavy Rain"]["registered"].sum()



#weather statistics
index=["Clear","Mist","Snow","Rain"]
values1=[Clear_registered,Mist_registered,Snow_registered,Rain_registered]
values2=[Clear_casual,Mist_casual,Snow_casual,Rain_casual]
plt.bar(index,values1,color='blue')
plt.bar(index,values2,color='red',bottom=values1)
plt.title("Weather statistics bar chart ")
plt.legend(["number of registered user ","number of non-registered user"])
plt.show()

clear_data=dataframe[dataframe["weather"]=="Clear"]
mist_data=dataframe[dataframe["weather"]=="Mist + Cloudy"]
snow_data=dataframe[dataframe["weather"]=="Light Snow"]
rain_data=dataframe[dataframe["weather"]=="Heavy Rain"]

# hours statistics with weather
y1=clear_data.groupby('hour')['count'].sum()
y2=mist_data.groupby('hour')['count'].sum()
y3=snow_data.groupby('hour')['count'].sum()
#y4=rain_data.groupby('hour')['count'].sum()  Only one line
y4=pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,164,0,0,0,0,0],range(24),)
x1=range(24)

plt.plot(x1,y1,'r',x1,y2,'b',x1,y3,'g',x1,y4,'y')
plt.title("Hour statistics with weather ")
plt.legend(["number of user in clear","number of user in mist","number of user in snow","number of user in rain"])
plt.show()

#month statistics
z1=clear_data.groupby('month')['count'].sum()
z2=mist_data.groupby('month')['count'].sum()
z3=snow_data.groupby('month')['count'].sum()
#z4=rain_data.groupby('month')['count'].sum() Only one line
z4=pd.DataFrame([164,0,0,0,0,0,0,0,0,0,0,0],range(12),)
x2=range(12)

plt.plot(x2,z1,'r',x2,z2,'b',x2,z3,'g',x2,z4,'y')
plt.title("Month statistics with weather ")
plt.legend(["user in clear","user in mist","user in snow","user in rain"])
plt.show()

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

#distribution
quartiles = pd.cut(dataframe["count"],10)
def get_stats(group):
    return {'counts': group.sum()}

grouped = dataframe["count"].groupby(quartiles)
count_distribution_amount = grouped.apply(get_stats).unstack()
df=pd.DataFrame(count_distribution_amount)
df.plot(kind='bar')
plt.title("count distribution")
plt.show()