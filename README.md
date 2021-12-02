# Stocks-Analysis-Tools
Prophet Stock Price Prediction
This information contained on this notebook and the resources avaiable for dowload through this website is not intended as, and shall not be understood or contruced as, financial advice!I developed this tool mainly to gain more experience in time series analysis and object-oriented programming. The goal is to combine different machine learning methods for stock analysis. I will occasionally update this project by adding more fuctions.
# Libraries for the project
* [Pandas](https://pandas.pydata.org)
* [Pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest)
* [Datetime](https://docs.python.org/3/library/datetime.html)
* [Prophet](https://facebook.github.io/prophet/docs/installation.html#installation-in-python)
* [Matplotlib](https://matplotlib.org)
# Results from the Prophet prediction
* Example of Stock Price Prediction

```
from metaprodictor import Prodictor

ticker = 'spy' 
days = 365 
test_ticker = Prodictor(ticker) 
test_ticker.predict(days)
```

![](image/stock_price_prediction.png)

* Example of Stock Trend Analysis

```
test_ticker.trend_analizer(365)
```

![](image/Porphet_Trend_analysis.png)

* Example of Stocks Daily Return Correlation Analysis
```
from metaprodictor import Corranalyzer

ticker_group = ['DVN','CLR','MRO','FANG','CVE','TRGP','SSL','EOG','COP','IMO']
highlight = 0.78

b = Corranalyzer(ticker_group)
b.mitrixCorrl()
```
![](image/CORRELATION.png)
