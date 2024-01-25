class Portfolio:
    def __init__(self, coins_list, initial_balance=0):
        self.balance = initial_balance

        self.amounts = {coin["symbol"] + '-USD': 0 for coin in coins_list}

    def buy(self, symbol, amount_spent, price_per_unit):
        amount_spent = min(amount_spent, self.balance)  # Ensure not to spend more than the available balance
        quantity = amount_spent / price_per_unit

        self.balance -= amount_spent
        self.amounts[symbol] += quantity

    def sell(self, symbol, amount_received, price_per_unit):
        amount_received = min(amount_received, self.amounts[symbol] * price_per_unit)  # Ensure not to sell more than available
        quantity_sold = amount_received / price_per_unit

        revenue = quantity_sold * price_per_unit
        self.balance += revenue
        self.amounts[symbol] -= quantity_sold
    
    def sell_all(self, prices):
        for symbol, amount in self.amounts.items():
            if amount > 0 and symbol in prices:
                self.sell(symbol, amount * prices[symbol], prices[symbol])
    
    def get_portfolio_value(self, prices):
        total_value = self.balance
        for symbol, amount in self.amounts.items():
            if symbol in prices:
                total_value += amount * prices[symbol]

        return total_value
