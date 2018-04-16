import pandas as pd
import matplotlib.pyplot as plt
dataframe=pd.read_csv("train.csv")



dataframe["hour"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[1].split(":")[0]))
dataframe["season"]=dataframe.season.map({1:"spring",2:"summer",3:"fall",4:"winter"})
dataframe["workingday"]=dataframe.workingday.map({0:"no",1:"yes"})
dataframe["weather"]=dataframe.weather.map({1:"Clear",2:"Mist + Cloudy",3:"Light Snow",4:"Heavy Rain"})

Clear_counts=dataframe[dataframe["weather"]=="Clear"]
Mist_counts=dataframe[dataframe["weather"]=="Mist + Cloudy"]
Snow_counts=dataframe[dataframe["weather"]=="Light Snow"]
Rain_counts=dataframe[dataframe["weather"]=="Heavy Rain"]


Clear_casual=Clear_counts["casual"].sum()
Mist_casual=Mist_counts["casual"].sum()
Snow_casual=Snow_counts["casual"].sum()
Rain_casual=Rain_counts["casual"].sum()

Clear_registered=Clear_counts["registered"].sum()
Mist_registered=Mist_counts["registered"].sum()
Snow_registered=Snow_counts["registered"].sum()
Rain_registered=Rain_counts["registered"].sum()




index=["Clear","Mist","Snow","Rain"]
values1=[Clear_registered,Mist_registered,Snow_registered,Rain_registered]
values2=[Clear_casual,Mist_casual,Snow_casual,Rain_casual]
plt.bar(index,values1,color='blue')
plt.bar(index,values2,color='red',bottom=values1)
plt.title("Weather statistics bar chart ")
plt.legend(["number of registered user ","number of non-registered user"])
plt.show()



