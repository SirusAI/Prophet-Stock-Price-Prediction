import pandas as pd
import datetime
import pandas_datareader.data as web
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mplt

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
        self.splot = stock['Adj Close']
        oc = stock['Close'] / stock['Open'] - 1
        new_data = stock
        new_data['o_c'] = oc
        self.new_df = new_data
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
    
    def oTc_return(self): #open to Close
        mplt.rc('figure', figsize = (20, 7))
        style.use('ggplot')
        d_return = self.new_df.iloc[:,-1]
        mean = d_return.mean(axis = 0, skipna = True)
        sd = d_return.std(axis = 0, skipna = True)
        u_ban = mean + 1.96 * sd
        l_ban = mean - 1.96 * sd
        p = 0
        total_gain = 0
        for i in d_return:
            if i > 0:
                p = p + 1
                total_gain = total_gain + i
        n = 0
        total_losses = 0
        for x in d_return:
            if x < 0:
                n = n + 1
                total_losses = total_losses + x
        edge = 0
        if total_gain > abs(total_losses):
            edge = 'Better Be a Buyer'
        else:
            edge = 'Better Be a Seller'
        
        length = len(d_return)
        win_rate = p / length
        l_rate = n / length
        

        data = {self.symbol:['Standard_Deviation','Upper_Band','Mean','Lower_Band','Positive_Days','Negative_Days',
                      'Total Days','Win_Rate','Losing_rate',
                      "Total_gains",'Total_losses','Edge'],
                'Open to Close Stats':["{:.5%}".format(sd),"{:.5%}".format(u_ban),"{:.5%}".format(mean),"{:.5%}".format(l_ban),p,n,
                               len(d_return),"{:.2%}".format(win_rate),
                               "{:.2%}".format(l_rate),total_gain,total_losses,edge]}
        
        df = pd.DataFrame.from_dict(data)
        d_return.plot(label = 'Open_To_Close_Return')
        return df.style.hide_index()
        
    def cTc_return(self): # Close to Close %
        mplt.rc('figure', figsize = (20, 7))
        style.use('ggplot')
        d_return = self.splot/self.splot.shift(1) -1
        new_df = self.stock
        new_df['daily_return'] = new_df['y']/new_df['y'].shift(1) -1
        mean = new_df.mean(axis = 0, skipna = True)[1]
        sd = new_df.std(axis = 0, skipna = True)[1]
        u_ban = mean + 1.96 * sd
        l_ban = mean - 1.96 * sd
        p = 0
        total_gain = 0
        for i in d_return:
            if i > 0:
                p = p + 1
                total_gain = total_gain + i
        n = 0
        total_losses = 0
        for x in d_return:
            if x < 0:
                n = n + 1
                total_losses = total_losses + x
        edge = 0
        if total_gain > abs(total_losses):
            edge = 'Better Be a Buyer'
        else:
            edge = 'Better Be a Seller'
        
        length = len(new_df)
        win_rate = p / (length -1)
        l_rate = n / (length - 1)
        

        data = {self.symbol:['Standard_Deviation','Upper_Band','Mean','Lower_Band','Positive_Days','Negative_Days',
                      'Total Days','Win_Rate','Losing_rate',
                      "Total_gains",'Total_losses','Edge'],
                'Close To Close Stats':["{:.5%}".format(sd),"{:.5%}".format(u_ban),"{:.5%}".format(mean),"{:.5%}".format(l_ban),p,n,
                               len(new_df)-1,"{:.2%}".format(win_rate),
                               "{:.2%}".format(l_rate),total_gain,total_losses,edge]}
        
        df = pd.DataFrame.from_dict(data)
        d_return.plot(label = 'Close_To_Close_Return')
        return df.style.hide_index()

class Corranalyzer:
    
    def __init__(self, tickers):
        
        self.tickers = tickers
        
        if len(self.tickers) > 1:
            self.tickers = tickers
        else:
            print ("need more input")
            
        datacomp = web.DataReader(self.tickers,'yahoo', start = start, end = end)['Adj Close']
        self.pct_return = datacomp.pct_change()
                   
    def matrixCorrl (self,highlight):

        highlight = highlight
        
        corr = self.pct_return.corr()
        
        self.corr = corr.copy()
        
        return corr.style.applymap(lambda x: 'background-color : orange' if x>highlight and x!=1 else '')
    
    def matrix_csv(self):
        data_frame = self.pct_return.corr()
        return data_frame
        
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
        ranking['Upper_Bound'] = ranking['Risk']*1.96 + ranking['MeanReturn']
        ranking['Lower_Bound'] = ranking['MeanReturn'] - ranking['Risk']*1.96
        length = len(self.tickers)
        b = pd.DataFrame(self.pct_return).dropna()    
        tuplelist_gain = []
        for ele in range(length):
            p = 0
            total_gain = 0
            total_days = 0
            for x in b.iloc[:,ele]:
                total_days = len(b.iloc[:,ele])
                if x > 0:
                    p = p + 1
                    total_gain = total_gain + x
            tuple = ([b.columns[ele],p,total_gain])
            tuplelist_gain.append(tuple)
        tuplelist_gain = pd.DataFrame(tuplelist_gain,columns=['Tickers','WinDays','Profit'])

        tuplelist_loss = []
        for ele in range(length):
            total_losses = 0
            n = 0
            total_days = 0
            for x in b.iloc[:,ele]:
                total_days = len(b.iloc[:,ele])
                if x < 0:
                    n = n + 1
                    total_losses = total_losses + x
            tuple = ([b.columns[ele],n,total_losses,total_days])
            tuplelist_loss.append(tuple)
        tuplelist_loss = pd.DataFrame(tuplelist_loss,columns=['Tickers','LossDays','Loss','TotalDays'])
        new_tablea = pd.DataFrame.merge(tuplelist_gain,tuplelist_loss)
        new_tablea['PnL'] = tuplelist_gain['Profit'] + tuplelist_loss['Loss']
        new_table = pd.DataFrame.merge(ranking,new_tablea)
        return new_table
