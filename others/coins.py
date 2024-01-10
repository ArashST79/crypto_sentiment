import requests

def get_top_coins(num):
    url = 'https://api.coinranking.com/v2/coins'

    headers = {
        'x-access-token': 'coinranking50375163ff48a0952b9f613314ed89000601da69a723cc83'
    }

    params = {'tiers': ['1']}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        coins_list = data['data']['coins'][:num]  
        return coins_list
    else:
        print(f"Request failed with status code: {response.status_code}")
        return []


num_of_coins = 10  
top_coins = get_top_coins(num_of_coins)

# for coin in top_coins:
#     print(f"Name: {coin['name']}, Symbol: {coin['symbol']}")
