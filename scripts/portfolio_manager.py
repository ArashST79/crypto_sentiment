import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self, coin_list, analyzed_data):
        self.coin_list = coin_list
        self.analyzed_data = analyzed_data

    def getWeights(self, current_date, lag_days=1):
        scores = {}
        for coin in self.coin_list:
            coin_data = self.analyzed_data[
            self.analyzed_data['related_coins'].apply(lambda coins: coin["symbol"] in coins or coin["name"] in coins)
            ]
            if(coin_data.empty): continue
            coin_data['Day'] = pd.to_datetime(coin_data['Day'])
            coin_data = coin_data[coin_data['Day'] <= current_date].sort_values('Day')
            avg_sent = coin_data.groupby(['Day']).agg({'sentiment' : ['mean', 'count']})
            avg_sent.columns = ['_'.join(str(i) for i in col) for col in avg_sent.columns]
            avg_sent.reset_index(inplace=True)
            alpha = 0.1  # Adjust the alpha for your desired rate of decay
            avg_sent['EMA'] = avg_sent['sentiment_mean'].ewm(alpha=alpha, adjust=False).mean()
            some_days_before = current_date - pd.DateOffset(days=lag_days)
            matching_rows = avg_sent[avg_sent['Day'].dt.date == some_days_before.date()]
            if not matching_rows.empty:
                score = matching_rows['EMA'].iloc[0]
            else:
                score = None  # or some other default value, depending on your needs

            coin_symbol = coin["symbol"]
            if(score == None): scores[coin_symbol] = 0
            else: scores[coin_symbol] = score

        top_coins = sorted(scores, key=scores.get, reverse=True)[:3]

        total_score = sum(scores[coin] for coin in top_coins)
        weights = {coin: scores[coin] / total_score for coin in top_coins}
        weights = {key + '-USD': value*100 for key, value in weights.items()}
        return weights

    def portfolio_dynamic_calculator(self,data, portfolio):
        balance_list = []
        for index, day in data.iterrows():
            day["Date"] = pd.to_datetime(day["Date"])
            weightings = self.getWeights(day["Date"])
            print(weightings)
            portfolio.sell_all(day)
            balance_list.append(portfolio.balance)
            print(portfolio.balance)
            for symbol, weight in weightings.items():
                amount = portfolio.balance*weight
                portfolio.buy(symbol=symbol,amount_spent=amount,price_per_unit=day[symbol])
            
        return balance_list
    
    def portfolio_static_calculator(self,weightings, data, portfolio):
        balance_list = []
        for index, day in data.iterrows():
            day["Date"] = pd.to_datetime(day["Date"])
            portfolio.sell_all(day)
            balance_list.append(portfolio.balance)
            for symbol, weight in weightings.items():
                amount = portfolio.balance*weight
                portfolio.buy(symbol=symbol,amount_spent=amount,price_per_unit=day[symbol])

        return balance_list
    
def get_averageindex_weights(coins_list):
    average_weights = {}
    for coin in coins_list:
        coin_symbol = coin['symbol']
        average_weights[coin_symbol] = 1/len(coins_list)

    average_weights = {key + '-USD': value*100 for key, value in average_weights.items()}
    return average_weights

