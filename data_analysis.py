import numpy as np
import pandas as pd
import json
import datetime
import spacy
from tqdm import tqdm
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px




class DataOrganize:
    def __init__(self, data):
        self.data = data

    def filter_by_date(self, start_date, end_date):
        self.data = self.data [(self.data ['created_at'] >= start_date) & (self.data ['created_at'] <= end_date)]


    def filter_by_coin(self, coin_names, coin_symbols):
        name_condition = self.data['body'].str.contains('|'.join(coin_names), case=False)
        symbol_condition = self.data['body'].str.contains('|'.join(coin_symbols), case=False)

        self.data = self.data[name_condition | symbol_condition]
        def find_related_coins(text):
            related_coins_for_row = []
            for name in coin_names:
                if name.lower() in text.lower():
                    related_coins_for_row.append(name)
            for symbol in coin_symbols:
                if symbol.lower() in text.lower():
                    related_coins_for_row.append(symbol)
            return related_coins_for_row
        self.data['related_coins'] = self.data['body'].apply(find_related_coins)


    def clean_data(self):
        raw_data = self.data
        def get_lang_detector(nlp, name):
            return LanguageDetector()

        nlp = spacy.load("en_core_web_sm")

        if not "language_detector" in Language.factories:
            Language.factory("language_detector", func=get_lang_detector)

        nlp.add_pipe('language_detector', last=True)

        def extract_features(x):
            doc = nlp(x)
            lang_dict = doc._.language
            language = lang_dict['language']
            entities = [ent.text for ent in doc.ents]
            return language, entities

        raw_data['language'], raw_data['entities'] = zip(*[extract_features(str(x)) for x in tqdm(raw_data['body'])])
        df_eng = raw_data[raw_data.language.values == 'en']
        df_new = df_eng.filter(items = ['id','body','created_at','user.followers','entities.sentiment.basic'])

        def parse_date(x):
            date_time_obj = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
            return date_time_obj.date().strftime('%Y-%m-%d')
        print(type(df_new["created_at"][0]))
        df_new['Day'] = [parse_date(x) for x in tqdm(df_new['created_at'])]
        self.data = df_new

    def analyze_sentiment(self):
        sid_obj = SentimentIntensityAnalyzer()
        def Vader_senti(x):
            """
            Function to calculate the sentiment of the message x.
            Returns the probability of a given input sentence to be Negative, Neutral, Positive and Compound score.
            
            """
            scores = sid_obj.polarity_scores(x)
            return scores['neg'],scores['neu'],scores['pos'],scores['compound']
        self.data [['vader_neg','vader_neu','vader_pos','vader_compound']] = [Vader_senti(x) for x in tqdm(self.data ['body'])]

