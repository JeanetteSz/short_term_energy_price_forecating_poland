# Short-term energy price forecasting in Poland

## Summary 
Poland is undergoing an energy transformation. More and more businesses and homes have their own photovoltaic installations and energy storage facilities. Proper energy management allows for optimized energy costs. Predicting energy prices in advance allows for planning appropriate energy use throughout the day. 
The project is focus on developing energy price forecasting model based on LSTM (Long Short-Term Memory) architecture.

## The dataset
The dataset contains hourly observations from January 2023 to May 2024. The dataset was divided into a train dataset and a test dataset according to date. The train dataset includes observations from January 2022 to December 2022, the test dataset from January 2023 to May 2023. 
<br>The dataset contains multiple variables to prediction energy price such as:
-	energy demand (source: https://energy.instrat.pl/en/electrical-system/load/)
-	wind energy generation (source: https://energy.instrat.pl/en/more-data/)
-	photovoltaic energy generation (source: https://energy.instrat.pl/en/more-data/)
-	carbon permits prices (source: https://energy.instrat.pl/en/prices/eu-ets/)
-	gas prices (source: https://energy.instrat.pl/en/prices/gas-dam/)
-	coal prices (https://energy.instrat.pl/en/mining/coal-prices-production-cost/ )
<br>The dataset also include variables as results of feature engineering process:
-	hourly peak
-	holidays
-	cyclical time-based features
It was consider including other features like weather data but to establish the most importance features for forecasting energy price, the correlation matrix was used. The results of correlation matrix indicate that weather data like temperature or rain has low impact on energy price. Wind speed has high correlation with energy price, but this information is included in feature – wind generation [MWh]. 

<img width="945" height="827" alt="image" src="https://github.com/user-attachments/assets/c4bd145b-2028-4e2f-a1b3-eb73761a0f40" />

## Seasonality in data
Energy prices are daily seasonality. Energy prices are different during the day and at the night. Moreover, during the day, energy prices increase at specific hours of the day. This is related to the increase in energy demand at this time. 

<img width="945" height="518" alt="image" src="https://github.com/user-attachments/assets/a32847a4-17b9-4d22-a9f0-9cd1dad99d16" />

## Technology
-	Neural Network (LSTM)
-		
The energy price forecasting model is based on LSTM, a type  of recurrent neural network (RNN). This type of RNN is especially recommended to prediction time series problems. While training model, Time Series Cross Validation was used to avoid overfitting. Process training of model is ready-to-use and was prepared according to Python OOP Concepts.

## Results
MSE	280.14
MAE	13.15
RMSE	16.74






## Plans for the development of project:	
-	Time forecasting: 48 hours
-	Hyper parameter optimalization with Optuna 
-	Integration MLFlow to python script
-	Tests

## How to run training the model?


## How to run testing the model?

