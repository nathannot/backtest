import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

class Backtesting:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end

    def get_data(self):
        x = yf.download(self.ticker,self.start, self.end, progress=False)[['Adj Close']]
        return x
    
    def generate_signal_sma(self, window):
        df = self.get_data()
        df = df.copy()
        df['sma'] = df['Adj Close'].rolling(window).mean()
        df['signal'] = (df['Adj Close']> df['sma']).astype(int)
        df.loc[(df['signal']==1)&(df['signal'].shift(1)==0),'signal'] = 2
        df.loc[(df['signal']==0)&(df['signal'].shift(1)==1),'signal'] = -1
        return df
    
    def generate_signal_ma_cross(self):
        df = self.get_data()
        df = df.copy()
        df['sma20'] = df['Adj Close'].rolling(20).mean()
        df['sma50'] = df['Adj Close'].rolling(50).mean()
        df['signal'] = 0
        df.loc[(df['sma20']>df['sma50'])&(df['sma20'].shift(1)<df['sma50'].shift(1)),'signal'] = 1
        df.loc[(df['sma50']>df['sma20'])&(df['sma20'].shift(1)>df['sma50'].shift(1)),'signal'] = -1
        return df
    
    def generate_signal_rsi(self, window):
        df = self.get_data()
        rets = df.diff().dropna()
        gain = rets.where(rets>0,0)
        loss = -rets.where(rets<0,0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        n = len(avg_gain)
        gain_s = np.zeros(n)
        loss_s = np.zeros(n)

        for i in range(14,n):
            gain_s[i] = ((n-1)*avg_gain.values[i-1][0]+gain.values[i][0])/n
            loss_s[i] = ((n-1)*avg_loss.values[i-1][0]+loss.values[i][0])/n
        rs = gain_s/loss_s
        rsi = 100-100/(1+rs)
        rsi = pd.Series(rsi, index = rets.index)
        df['rsi'] = rsi
        df['signal']=0
        df.loc[(df['rsi']>30)&(df['rsi'].shift(1)<30),'signal'] = 1
        df.loc[(df['rsi']>80)&(df['rsi'].shift(1)<80),'signal'] = -1
        return df
    
    def generate_signal_bbands(self, window):
        df = self.get_data()
        roll_mean = df['Adj Close'].rolling(window).mean()
        roll_std = df['Adj Close'].rolling(window).std()
        df['bband_upper'] = roll_mean+2*roll_std
        df['bband_lower'] = roll_mean - 2*roll_std
        df['signal'] = 0
        df.loc[(df['Adj Close']>df['bband_lower'])&
               (df['Adj Close'].shift(1)<df['bband_lower'].shift(1)),'signal'] = 1
        df.loc[(df['Adj Close']>df['bband_upper'])&
               (df['Adj Close'].shift(1)<df['bband_upper'].shift(1)), 'signal'] = -1
        return df
    
    def execute_trade_sma(self, initial, weight, tc, window):
        x = self.generate_signal_sma(window)
        tc = tc/100
        shares = 0
        cash = initial*weight
        position = 0
        history = []
        ports = []
        for index, row in x.iterrows():
            date = index
            price = row['Adj Close']
            signal = row['signal']
    
            if signal == 2 and position == 0:
                new_shares = cash//(price*(1+tc))
                cash -= new_shares*price*(1+tc)
                shares += new_shares
                t_cost = tc*price*new_shares
                position = 1
                port = cash+shares*price
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'buy',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            elif signal == -1 and position ==1:
                cash += shares*price*(1-tc)
                t_cost = shares*price*tc
                shares = 0
                position = 0
                port = cash + price*shares
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'sell',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            
            full_port = cash+shares*price
            ports.append(full_port)
            
        return ports, pd.DataFrame(history)
    
    def execute_trade_rsi(self,initial, weight, tc, window):
        x = self.generate_signal_rsi(window)
        tc = tc/100
        cash = weight*initial
        shares = 0
        history = []
        ports = []
        position = 0

        for index, row in x.iterrows():
            date = index
            price = row['Adj Close']
            signal = row['signal']

            if signal == 1 and position == 0:
                num_shares = cash // (price*(1+tc))
                cash -= num_shares*price*(1+tc)
                shares += num_shares
                t_cost = tc*price*num_shares
                position = 1
                port = cash+shares*price
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'buy',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            elif signal == -1 and position == 1:
                cash += shares*price*(1-tc)
                t_cost = shares*(price*tc)
                shares = 0
                position = 0
                port = cash
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'sell',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            full_port = cash+price*shares
            ports.append(full_port)
        return ports, pd.DataFrame(history)
    
    def execute_trade_bbands(self, initial, weight, tc, window):
        x = self.generate_signal_bbands(window)
        cash = weight*initial
        tc = tc/100
        shares = 0
        position = 0
        history = []
        ports = []
        for index, row in x.iterrows():
            date = index
            price = row['Adj Close']
            signal = row['signal']
            if signal == 1 and position == 0:
                num_shares = cash // (price*(1+tc))
                cash -= num_shares*price*(1+tc)
                t_cost = num_shares*price*tc
                position = 1
                shares += num_shares
                port = cash + shares*price
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'buy',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            elif signal == -1 and position == 1:
                cash += shares*price*(1-tc)
                t_cost = shares*(price*tc)
                shares = 0
                position = 0
                port = cash
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'sell',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            full_port = cash+shares*price
            ports.append(full_port)
        return ports, pd.DataFrame(history)

    def execute_trade_ma_cross(self,initial, weight, tc):
        x = self.generate_signal_ma_cross()
        tc = tc/100
        cash = weight*initial
        shares = 0
        history = []
        ports = []
        position = 0

        for index, row in x.iterrows():
            date = index
            price = row['Adj Close']
            signal = row['signal']

            if signal == 1 and position == 0:
                num_shares = cash // (price*(1+tc))
                cash -= num_shares*price*(1+tc)
                shares += num_shares
                t_cost = tc*price*num_shares
                position = 1
                port = cash+shares*price
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'buy',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            elif signal == -1 and position == 1:
                cash += shares*price*(1-tc)
                t_cost = shares*(price*tc)
                shares = 0
                position = 0
                port = cash
                history.append({'Date':date,
                                'Price':price,
                                'Type': 'sell',
                                'shares': shares,
                                'Portfolio':port,
                                'Transaction cost':t_cost})
            full_port = cash+price*shares
            ports.append(full_port)
        return ports, pd.DataFrame(history)

    def buy_and_hold(self, initial, weight, tc):
        df = self.get_data()
        df['bnh'] = 0
        df.loc[df.index[0], 'bnh'] = 1
        df.loc[df.index[-1], 'bnh'] = -1
        tc = tc/100
        cash = weight*initial
        position = 0
        shares = 0
        hist = []
        bench = []
        for index, row in df.iterrows():
            date = index
            price = row['Adj Close']
            signal = row['bnh']
            if signal == 1 and position == 0:
                new_shares = cash //(price*(1+tc))
                cash -= new_shares*price*(1+tc)
                shares += new_shares
                position = 1
                port = cash + shares*price
            elif signal == -1 and position == 1:
                cash += shares*price*(1-tc)
                shares = 0
                position = 0
                port = cash
            full_port = cash + shares*price
            bench.append(full_port)
        return bench
        

    def port_plot_sma(self, initial, weight, tc, window):
        x = self.generate_signal_sma(window)
        bench = self.buy_and_hold(initial, weight, tc)
        port, _ = self.execute_trade_sma(initial, weight, tc, window)
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        ax[0].plot(x.index, port, label = f'{strat}')
        ax[0].plot(x.index, bench, label='Buy and Hold')
        ax[0].legend()
        ax[0].set_title('Portfolio Value')
        ax[0].grid()
        ax[1].plot(x['Adj Close'])
        ax[1].plot(x['sma'], label='20 day moving average')
        ax[1].set_title(f'Price of {ticker}')
        ax[1].legend()
        ax[1].grid()
        plt.tight_layout()
        plt.show()
        return fig
    
    def port_plot_cross_ma(self, initial, weight, tc):
        x = self.generate_signal_ma_cross()
        bench = self.buy_and_hold(initial, weight, tc)
        port, _ = self.execute_trade_ma_cross(initial, weight, tc)
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        ax[0].plot(x.index, port, label = f'{strat}')
        ax[0].plot(x.index, bench, label='Buy and Hold')
        ax[0].legend()
        ax[0].set_title('Portfolio Value')
        ax[0].grid()
        ax[1].plot(x['Adj Close'])
        ax[1].plot(x['sma20'], label='20 day moving average')
        ax[1].plot(x['sma50'], label='50 day moving average')
        ax[1].set_title(f'Price of {ticker}')
        ax[1].legend()
        ax[1].grid()
        plt.tight_layout()
        plt.show()
        return fig

    def port_plot_rsi(self, initial, weight, tc, window):
        x = self.generate_signal_rsi(window)
        bench = self.buy_and_hold(initial, weight, tc)
        port, _ = self.execute_trade_sma(initial, weight, tc, window)
        fig, ax = plt.subplots(3,1, figsize=(8,8))
        ax[0].plot(x.index, port, label = f'{strat}')
        ax[0].plot(x.index, bench, label='Buy and Hold')
        ax[0].legend()
        ax[0].grid()
        ax[0].set_title('Portfolio Value')
        ax[1].plot(x['Adj Close'])
        ax[1].set_title(f'Price of {ticker}')
        ax[1].grid()
        ax[2].plot(x['rsi'])
        ax[2].axhline(y=70, xmin=0, xmax=len(x)-1, color='r', linestyle='--')
        ax[2].axhline(y=30, xmin=0, xmax=len(x)-1, color='r', linestyle='--')
        ax[2].set_title('14 day RSI')
        ax[2].grid()
        plt.tight_layout()
        plt.show()
        return fig
    
    def port_plot_bbands(self, initial, weight, tc, window):
        x = self.generate_signal_bbands(window)
        bench = self.buy_and_hold(initial, weight, tc)
        port, _ = self.execute_trade_sma(initial, weight, tc, window)
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        ax[0].plot(x.index, port, label = f'{strat}')
        ax[0].plot(x.index, bench, label='Buy and Hold')
        ax[0].legend()
        ax[0].set_title('Portfolio Value')
        ax[0].grid()
        ax[1].plot(x['Adj Close'])
        ax[1].plot(x['bband_upper'], color='r',lw=0.3, label='Upper Bollinger Band')
        ax[1].plot(x['bband_lower'], color='r',lw=0.3, label='Lower Bollinger Band')
        ax[1].set_title(f'Price of {ticker}')
        ax[1].grid()
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        return fig

st.header('Backtesting Strategy for Stocks')
ticker = st.selectbox('Choose from following stocks',
                      ('aapl','tsla','goog','nvda','meta',
                       'amzn','nflx'))

start = st.date_input('Select start date', value=datetime(2023,1,1))
finish = st.date_input('Select end date',min_value = start+pd.Timedelta(days=126))
data = Backtesting(ticker, start, finish)

strat = st.selectbox('Select from following strategies',
                     ('20 Day Moving Average','14 Day RSI',
                      'Bollinger bands', '20-50 Day Moving Average Cross'))

tran_cost = st.slider('Select transaction cost %', value=0.5, min_value=0.0, max_value = 5.0, step=0.1)
st.write(f'Backtest $10000 portfolio for {ticker}')
if strat == '20 Day Moving Average':
    port, hist = data.execute_trade_sma(10000,1,tran_cost,20)
    fig = data.port_plot_sma(10000,1,tran_cost,20)
    st.pyplot(fig)
    st.write('This table summarises all trades.')

    st.write(hist)
elif strat == '14 Day RSI':
    port, hist = data.execute_trade_rsi(10000,1,tran_cost,14)
    fig = data.port_plot_rsi(10000,1,tran_cost,14)
    st.pyplot(fig)
    st.write('This table summarises all trades.')
    st.write(hist)
elif strat == 'Bollinger bands':
    port, hist = data.execute_trade_bbands(10000,1,tran_cost,14)
    fig = data.port_plot_bbands(10000,1,tran_cost,14)
    st.pyplot(fig)
    st.write('This table summarises all trades.')
    st.write(hist)
elif strat == '20-50 Day Moving Average Cross':
    port, hist = data.execute_trade_ma_cross(10000,1,tran_cost)
    fig = data.port_plot_cross_ma(10000,1,tran_cost)
    st.pyplot(fig)
    st.write('This table summarises all trades.')
    st.write(hist)

