import pandas as pd
import datetime
import pandas_datareader.data as web
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mplt
import matplotlib
import fbprophet

start = datetime.datetime(2005, 1, 1)
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
        new_datas = stock.reset_index(level=0)
        self.new_datas = new_datas
    
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
        edge = []
        if total_gain > abs(total_losses):
            edge = 'Better Be a Buyer'
        else:
            edge = 'Better Be a Seller'
        
        length = len(d_return)
        win_rate = p / length
        l_rate = n / length

        max_drawdown = d_return.min()
        max_return = d_return.max()
        data = {self.symbol:['Standard_Deviation','Upper_Band','Mean','Lower_Band','Max_drawdown','Max_gain','Positive_Days','Negative_Days',
                      'Total Days','Win_Rate','Losing_rate',
                      'Total_gains','Total_losses','PnL','Edge'],
                'Open to Close Stats':["{:.5%}".format(sd),'{:.5%}'.format(u_ban),'{:.5%}'.format(mean),'{:.5%}'.format(l_ban),
                                       '{:.5%}'.format(max_drawdown),"{:.5%}".format(max_return),p,n,
                               len(d_return),'{:.2%}'.format(win_rate),
                               '{:.2%}'.format(l_rate),total_gain,total_losses,total_gain+total_losses,edge]}
        
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
        edge = []
        if total_gain > abs(total_losses):
            edge = 'Better Be a Buyer'
        else:
            edge = 'Better Be a Seller'
        length = len(new_df)
        win_rate = p / (length -1)
        l_rate = n / (length - 1)
        max_drawdown = new_df['daily_return'].min()
        max_return = new_df['daily_return'].max()
        data = {self.symbol:['Standard_Deviation','Upper_Band','Mean','Lower_Band','Max_drawdown','Max_gain','Positive_Days','Negative_Days',
                      'Total Days','Win_Rate','Losing_rate',
                      'Total_gains','Total_losses','PnL','Edge'],
                'Close To Close Stats':['{:.5%}'.format(sd),'{:.5%}'.format(u_ban),'{:.5%}'.format(mean),'{:.5%}'.format(l_ban),
                                        '{:.5%}'.format(max_drawdown),'{:.5%}'.format(max_return),p,n,
                               len(new_df)-1,'{:.2%}'.format(win_rate),
                               '{:.2%}'.format(l_rate),total_gain,total_losses,total_gain+total_losses,edge]}
        
        df = pd.DataFrame.from_dict(data)
        d_return.plot(label = 'Close_To_Close_Return')
        return df.style.hide_index()
    
    def zTc_return(self,days): # Close to Close %
        days = days
        mplt.rc('figure', figsize = (20, 7))
        style.use('ggplot')
        new_df = self.stock
        a=[]
        b=[]
        for i in range(days):
            b = new_df['y']/new_df['y'].shift(i+1) -1
            c = b.iloc[1::i+1].mean()
            d = b.iloc[1::i+1].std()
            tuple = (i+1,c,d)
            a.append(tuple)
        a = sorted(a, key=lambda item: item[1],reverse=True)
        optimized_number = [item[0] for item in a][0]
        optimized_mean = [item[1] for item in a][0]
        optimized_sd = [item[2] for item in a][0]
#         return optimized_sd
        mean = optimized_mean
        sd = optimized_sd
        u_ban = mean + 1.96 * sd
        l_ban = mean - 1.96 * sd
        
        p = 0
        total_gain = 0
        d_return = new_df['y']/new_df['y'].shift(optimized_number) -1
        
        for i in d_return.iloc[1::optimized_number]:
            if i > 0:
                p = p + 1
                total_gain = total_gain + i
        n = 0
        total_losses = 0
        for x in d_return.iloc[1::optimized_number]:
            if x < 0:
                n = n + 1
                total_losses = total_losses + x
        edge = []
        if total_gain > abs(total_losses):
            edge = 'Better Be a Buyer'
        else:
            edge = 'Better Be a Seller'

        c = d_return.iloc[1::optimized_number]
        win_rate = p / len(c)
        l_rate = n / len(c)
        max_drawdown = c.min()
        max_return = c.max()
        data = {self.symbol:['Best Holiding Periods(Trading Days)','Standard_Deviation','Upper_Band','Mean Return','Lower_Band',
                             'Max_drawdown','Max_gain','Winning_Trades','Lossing_Trades',
                      'Total_trade','Win_Rate','Losing_rate',
                      'Total_gains','Total_losses','PnL','Total_Trading_Days','Edge'],
                'Best Trading Horizon':[optimized_number,'{:.5%}'.format(sd),'{:.5%}'.format(u_ban),'{:.5%}'.format(mean),
                                        '{:.5%}'.format(l_ban), '{:.5%}'.format(max_drawdown),"{:.5%}".format(max_return),p,n,
                               len(c),'{:.2%}'.format(win_rate),
                               '{:.2%}'.format(l_rate),total_gain,total_losses,total_gain+total_losses,len(new_df),edge]}
        
        df = pd.DataFrame.from_dict(data)
        bpp  = self.splot/self.splot.shift(optimized_number) -1
        bpp.plot(label = 'Close_To_Close_Return')
        return df.style.hide_index()

    def model(self):

        # Make the model
        monthly_seasonality = True
        model = fbprophet.Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            changepoints=None,)

        if monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        return model
    
    def turningPoint(self, search=None):
        matplotlib.rcdefaults()
        matplotlib.rcParams['figure.figsize'] = (12, 7)
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 12
        matplotlib.rcParams['ytick.labelsize'] = 12
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
        model = self.model()
        train_data = self.new_datas
        train_data['ds'] = train_data['Date']
        min_date = min(train_data['Date'])
        max_date = max(train_data['Date'])
        years = 20
        if 'Adj. Close' not in train_data.columns:
            train_data['Adj. Close'] = train_data['Close']
            train_data['Adj. Open'] = train_data['Open']

        train_data['y'] = train_data['Adj. Close']
        train_data['Daily Change'] = train_data['Adj. Close'] - train_data['Adj. Open']
        # training years  = 20 years
        train = train_data[train_data['Date']> (max_date - pd.DateOffset(years=years))]
        model.fit(train)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)
        train = pd.merge(train, future[['ds', 'yhat']], on='ds', how='inner')
        changepoints = model.changepoints
        train = train.reset_index(drop=True)
        change_indices = []
        for changepoint in changepoints:
            change_indices.append(train[train['ds'] == changepoint].index[0])
        c_data = train.loc[change_indices, :]
        deltas = model.params['delta'][0]
        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])
        c_data = c_data.sort_values(by='abs_delta', ascending=False)
        c_data = c_data[:10]
        cpos_data = c_data[c_data['delta'] > 0]
        cneg_data = c_data[c_data['delta'] < 0]
        if not search:
            plt.plot(train['ds'], train['y'], 'ko', ms=4, label='Stock Price')
            plt.plot(future['ds'],future['yhat'],color='cyan',linewidth=2.0,label='Modeled',)

            plt.vlines(cpos_data['ds'].dt.to_pydatetime(),ymin=min(train['y']),ymax=max(train['y']),linestyles='dashed',
                color='r',linewidth=1.2,label='Negative Changepoints',)

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(),ymin=min(train['y']),ymax=max(train['y']),linestyles='dashed',
                color='darkgreen',linewidth=1.2,label='Positive Changepoints',)

            plt.legend(prop={'size': 10})
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.title('Stock Price with Changepoints')
            plt.show()
            print('\nChangepoints sorted by slope rate of change:\n')
            return c_data.loc[:, ['Date', 'Adj. Close', 'delta']][:5].style.hide_index()

        if search:
            date_range = ['%s %s' % (str(min(train['Date'])), str(max(train['Date'])))]
            trends, related_queries = self.retrieve_google_trends(search, date_range)
            if (trends is None) or (related_queries is None):
                print('No search trends found for %s' % search)
                return
            print('\n Top Related Queries: \n')
            print(related_queries[search]["top"].head())
            print('\n Rising Related Queries: \n')
            print(related_queries[search]['rising'].head())
            trends = trends.resample('D')
            trends = trends.reset_index(level=0)
            trends = trends.rename(columns={'date': 'ds', search: 'freq'})
            trends['freq'] = trends['freq'].interpolate()
            train = pd.merge(train, trends, on="ds", how='inner')
            train['y_norm'] = train['y'] / max(train['y'])
            train['freq_norm'] = train['freq'] / max(train['freq'])
            self.reset_plot()
            plt.plot(train['ds'], train['y_norm'], 'k-', label='Stock Price')
            plt.plot(train['ds'],train['freq_norm'],color='goldenrod',label='Search Frequency',)
            plt.vlines(cpos_data['ds'].dt.to_pydatetime(),ymin=0,ymax=1,linestyles='dashed',color='r',linewidth=1.2,
                       label='Negative Changepoints',)

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(),ymin=0,ymax=1,linestyles='dashed',color='darkgreen',linewidth=1.2,
                label='Positive Changepoints',)

            plt.legend(prop={'size': 10})
            plt.xlabel('Date')
            plt.ylabel('Normalized Values')
            plt.title('%s Stock Price and Search Frequency for %s' % (self.symbol, search))
            plt.show()   
    
class Corranalyzer:
    
    def __init__(self, tickers):
        
        self.tickers = tickers
        
        if len(self.tickers) > 1:
            self.tickers = tickers
        else:
            print ('need more input')
            
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
        risk_ranking = pd.DataFrame(risk_ranking,columns=['Symbols', 'Risk'])
        return_ranking = sorted(self.pct_return.mean().items(), key=lambda item: item[1],reverse=True)
        return_ranking = pd.DataFrame(return_ranking, columns=['Symbols', 'MeanReturn'])
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
        tuplelist_gain = pd.DataFrame(tuplelist_gain,columns=['Symbols','WinDays','Profit'])

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
        tuplelist_loss = pd.DataFrame(tuplelist_loss,columns=['Symbols','LossDays','Loss','TotalDays'])
        new_tablea = pd.DataFrame.merge(tuplelist_gain,tuplelist_loss)
        new_tablea['PnL'] = tuplelist_gain['Profit'] + tuplelist_loss['Loss']
        new_table = pd.DataFrame.merge(ranking,new_tablea)
#         print('Total Trading Days: ',len(b))
        return new_table.style.hide_index()
