import pandas as pd
import datetime
import pandas_datareader.data as web
from prophet import Prophet

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today().strftime('%Y-%m-%d')

class Prodictor:
    def __init__ (self, ticker):

        ticker = ticker.upper()
        
        self.symbol = ticker #for future fuction
        
        try:
            stock = web.DataReader(ticker, 'yahoo', start, end)
        except Exception as e:
            print('Error Retrieving Data.')
            print (e)
            return
        df_a = stock.reset_index(col_level=0)
        df_a.rename(columns={'Date':'ds',"Adj Close":'y'}, inplace=True)
        df_a = df_a[['ds','y']]
        self.stock = df_a.copy()
    
    def predict(self,days):
        
        p = Prophet(daily_seasonality=False)
        
        p.fit(self.stock)
        
        future = p.make_future_dataframe(periods = days)
        
        forecast = p.predict(future)
        
        fig_a = p.plot(forecast,xlabel = self.symbol,ylabel = 'Price')
        
    
    def trend_analizer(self,days):
        
        p = Prophet(daily_seasonality=False)
        
        p.fit(self.stock)
        
        future = p.make_future_dataframe(periods = days)
        
        forecast = p.predict(future)

        fig_b = p.plot_components(forecast)
class Corranalyzer:
    
    def __init__(self, tickers):
        
        self.tickers = tickers
        
        if len(self.tickers) > 1:
            self.tickers = tickers
        else:
            print ("need more input")
        
    def matrixCorrl (self,highlight):
        
        highlight = highlight
        
        datacomp = web.DataReader(self.tickers,'yahoo', start = start, end = end)['Adj Close']
        
        pct_return = datacomp.pct_change()
                
        corr = pct_return.corr()
        
        self.corr = corr.copy()
        
        return corr.style.applymap(lambda x: 'background-color : orange' if x>highlight and x!=1 else '')

    # Ranking risk / mean return of the sotcks list
    def riskRank(self):
        plt.figure(figsize=(20,10),dpi = 75)
        plt.scatter(self.pct_return.mean(), self.pct_return.std())
        plt.xlabel('Expected returns')
        plt.ylabel('Risk')
        for label, x, y in zip(self.pct_return.columns, self.pct_return.mean(), self.pct_return.std()):
            plt.annotate(
                label, 
                xy = (x, y), xytext = (20, -20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        n = len(self.pct_return)
        risk_ranking = sorted(self.pct_return.std().items(), key=lambda item: item[1],reverse=True)
        risk_ranking = pd.DataFrame(risk_ranking,columns=['Tickers', 'Risk'])
        return_ranking = sorted(self.pct_return.mean().items(), key=lambda item: item[1],reverse=True)
        return_ranking = pd.DataFrame(return_ranking, columns=['Tickers', 'MeanReturn'])
        sd = risk_ranking['Risk']/(n**0.5)
        return_ranking['SD_Error'] = sd
        ranking = pd.DataFrame.merge(risk_ranking,return_ranking)
        print('Ranking: ')
        display(ranking)
