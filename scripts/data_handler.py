
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
from scripts.sentiment_analysor import Predictor
from utilities import Vader_senti
import glob
import pandas as pd
from enum import Enum

class DataModes(Enum):
    FULL_DATA = "full_data"
    FIRST_1000 = "first_1000"
    RANDOM_1000 = "random_1000"
    RANDOM_10000 = "random_10000"
    RANDOM_100000 = "random_100000"
    HALF_DATA = "half_data"
    EVAL_100000="eval_100000"
    EVAL_1000="eval_1000"

class DataOrganize:

    def __init__(self):
        self.data = None
    def create_clean_data(self, mode=DataModes.FULL_DATA):
        self.mode = mode
        files = glob.glob('../data/StockTwits.*_messages.csv')

        dfs = []
        columns_to_read = ['user.username', 'body', 'created_at', 'user.followers', 'entities.sentiment.basic']
        for file in files:
            df = pd.read_csv(file, nrows=100000,usecols=columns_to_read)
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data.drop_duplicates(subset=['user.username', 'body'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = data[data['created_at'].str.startswith('2023')]
        data.reset_index(drop=True, inplace=True)
        max_text_length = 300
        data = data[data['body'].str.len() <= max_text_length]
        data.reset_index(drop=True, inplace=True)

        if mode == DataModes.FIRST_1000:
            data = data.head(1000)
        elif mode == DataModes.RANDOM_1000:
            data = data.sample(n=1000, random_state=42) 
        elif mode == DataModes.RANDOM_10000:
            data = data.sample(n=10000, random_state=42) 
        elif mode == DataModes.RANDOM_100000:
            data = data.sample(n=100000, random_state=42) 
        elif mode == DataModes.HALF_DATA:
            data = data.sample(n=int(len(data)/2), random_state=42) 
        elif mode == DataModes.EVAL_100000:
            data2 = data.sample(n=100000, random_state=42)
            data = data.drop(data2.index)
        elif mode == DataModes.EVAL_1000:
            data2 = data.sample(n=100000, random_state=42)
            data = data.drop(data2.index)
            data = data.iloc[:1000]

        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.clean_data()
        return self.data

    def filter_by_date(self, start_date, end_date):
        self.data = self.data [(self.data ['created_at'] >= start_date) & (self.data ['created_at'] <= end_date)]


    def group_by_coin(self, coins_list):
        coins_name = [coin['name'] for coin in coins_list]
        coins_symbol = [coin['symbol'] for coin in coins_list]
        name_condition = self.data['body'].str.contains('|'.join(coins_name), case=False)
        symbol_condition = self.data['body'].str.contains('|'.join(coins_symbol), case=False)

        self.data = self.data[name_condition | symbol_condition]
        def find_related_coins(text):
            related_coins_for_row = []
            for name in coins_name:
                if name.lower() in text.lower():
                    related_coins_for_row.append(name)
            for symbol in coins_symbol:
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
            return date_time_obj.date()
        
        df_new['Day'] = [parse_date(x) for x in tqdm(df_new['created_at'])]
        print("day type :" + str(type(df_new['Day'][0])))
        self.data = df_new

    def coins_sentiment_apply(self,coins_list):
        self.group_by_coin(coins_list)
        from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
        from scripts.sentiment_analysor import Evaluator
        import torch
        model_name = "ElKulako/cryptobert"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load("../models/fine_tuned_cryptobert.pth",map_location=torch.device('cpu')))

        predictor = Predictor(model,tokenizer,self.data)

        predictor.predict()
        # Mapping bullish to +1 and bearish to -1
        sentiment_mapping = {"Bullish": 1, "Bearish": -1}

        # Applying the mapping to the list of sentiment labels
        integer_labels = [sentiment_mapping[label] for label in predictor.predicted_labels]

        # Assigning the list of integers to the 'sentiment' column in your DataFrame
        self.data['sentiment'] = integer_labels

