# Bikesharing
Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis. 

The data generated by these systems makes them attractive for researchers because the duration of travel, departure location, arrival location, and time elapsed is explicitly recorded. Bike sharing systems therefore function as a sensor network, which can be used for studying mobility in a city. [Data source:kaggle](https://www.kaggle.com/c/bike-sharing-demand/data)

This project makes use of  historical usage patterns and weather data to forecast amount of user

# Data Fields
* datetime - hourly date + timestamp 
* season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
* holiday - whether the day is considered a holiday
* workingday - whether the day is neither a weekend nor holiday
* weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
<br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
* temp - temperature in Celsius
* atemp - "feels like" temperature in Celsius
* humidity - relative humidity
* windspeed - wind speed
* casual - number of non-registered user rentals initiated
* registered - number of registered user rentals initiated
* count - number of total rentals

# Data visualization
* The distribution of the number of users per hour
![](https://github.com/suangzi123/bikesharing/blob/master/images/Count_distribution.png)<br>
A total of 10886 samples and the samples outside 3 std account for less than 1% of data.The outlier data will be filtered out.
* Correlation matrix
![](https://github.com/suangzi123/bikesharing/blob/master/images/Correlation_matrix.png)<br>
There are no obvious linear relationship among temperature,humidity,windspeed and user count.
* Hours mean statistics with weather
![](https://github.com/suangzi123/bikesharing/blob/master/images/Hour_statistics.png)<br>
 This picture show the different numbers of user count per hour in different weather.There will be more users in good weather and the morning peak is at eight in the morning and the evening peak is at seventeen in the evening
 * Month statistics
 
 ![](https://github.com/suangzi123/bikesharing/blob/master/images/Month_statistics.png)
 
 This picture show the mean of user count per hour in different month and different weather.January has the fewest users.
 
 * Season statistics
 
 ![](https://github.com/suangzi123/bikesharing/blob/master/images/Season_statistics.png)
 
  This picture show the user count in different season and different weather.There are more users in summer and fall and less in spring and winter.
  
  # Regression
  Choosing the appropriate model for regression by comparing the accuracy of different regression models.After trying multiple linear regression, poly linear regression, SVR, Decision Tree regression,Random Forest regression,XGRegressor and select optimal parameter by grid search,XGRegressor gives the maximum  k-fold R2 score [ 0.83592285  0.83862756  0.83806972  0.83190081  0.8472441 ]and minimum mean square error(4414.09082941)in testing set.
  * XGBoost regression predicted results in testing set
  
  ![](https://github.com/suangzi123/bikesharing/blob/master/images/XGBoost_Regression_predicted_results.png)
  
