import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self, coin_list, analyzed_data):
        self.coin_list = coin_list
        self.analyzed_data = analyzed_data

    def getWeights(self, current_date):
        scores = {}
        for coin in self.coin_list:
            coin_data = self.analyzed_data[
            self.analyzed_data['related_coins'].apply(lambda coins: coin["symbol"] in coins or coin["name"] in coins)
            ]
            if(coin_data.empty): continue
            coin_data['Day'] = pd.to_datetime(coin_data['Day'])
            coin_data = coin_data[coin_data['Day'] <= current_date].sort_values('Day')
            avg_sent = coin_data.groupby(['Day']).agg({'vader_compound' : ['mean', 'count']})
            avg_sent.columns = ['_'.join(str(i) for i in col) for col in avg_sent.columns]
            avg_sent.reset_index(inplace=True)
            alpha = 0.1  # Adjust the alpha for your desired rate of decay
            avg_sent['EMA'] = avg_sent['vader_compound_mean'].ewm(alpha=alpha, adjust=False).mean()
            one_day_before = current_date - pd.DateOffset(days=1)
            matching_rows = avg_sent[avg_sent['Day'].dt.date == one_day_before.date()]
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

        return weights
