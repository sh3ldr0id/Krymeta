import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    handlers=[
        logging.FileHandler(
            filename="logs.log",
            mode="w"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info("Importing Required Modules")

from json import loads

from binance.client import Client

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from numpy import diff, array

from datetime import timedelta, datetime
from time import sleep, localtime

class Krymeta:
    def __init__(self, 
                 api_key: str, 
                 api_secret: str,
                 base: str = "BTC", 
                 quote: str = "BUSD", 
                 amount: float = 20,
                 leverage: int = 5,
                 FREQUENCY: int = 20,
                 ACCURACY: float = 0.80
                ) -> None:
        self.base = base
        self.quote = quote

        self.symbol = base + quote

        self.amount = amount
        self.leverage = leverage

        self.FREQUENCY = FREQUENCY
    
        self.ACCURACY = ACCURACY

        logging.info(f"Trading Pair -> {self.symbol}")

        logging.info("Connecting to Binance API")

        self.client = Client(
            api_key=api_key,
            api_secret=api_secret
        )

        self.client.futures_change_leverage(
            symbol=self.symbol, 
            leverage=self.leverage
        )

        self.prices = []

        self.fetch_prices()

        self.features = None
        self.labels = None

        self.proccess_data()

        self.model = None

        self.create_model()

        self.rolling_sums = None

        self.initial_cycle()

        self.holding = False
        self.prev_balance = None
        self.last_buy = 0

        self.trade_cycle()

    def fetch_prices(self) -> None:
        logging.info("Fetching Historical Data")

        dataset = self.client.futures_historical_klines_generator(
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            start_str=str(datetime.today()-timedelta(
                days=20
            ))
        )

        for data in dataset:
            self.prices.append(
                float(
                    data[4]
                )
            )

    def proccess_data(self) -> None:
        logging.info("Processing Data")

        rolling_sums = []

        for index in range(len(self.prices)-5):
            rolling_sums.append(
                diff(
                    self.prices[index:index+5]
                ).sum()
            )

        actions = []

        for rolling_sum in rolling_sums:
            if rolling_sum > 0:
                actions.append(0)

            else:
                actions.append(1)

        logging.info("Converting to features and labels")

        features = []
        labels = []

        for index in range(self.FREQUENCY, len(rolling_sums)):
            features.append(
                rolling_sums[index-self.FREQUENCY:index]
            )

            labels.append(
                actions[index]
            )

        self.features = array(
            features
        ).reshape(-1, 1, self.FREQUENCY)

        self.labels = array(
            labels
        )

    def create_model(self) -> None:
        logging.info("Creating Model")

        model = Sequential()

        model.add(
            LSTM(
                units=50,
                input_shape=(
                    self.features.shape[1],
                    self.features.shape[2]
                ),
                return_sequences=True
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            LSTM(
                units=50,
                return_sequences=True
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            LSTM(
                units=25,
                return_sequences=False
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(
                units=2,
                activation="sigmoid"
            )
        )

        model.compile(
            optimizer="RMSprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logging.info("Fitting Model")

        model.fit(
            x=self.features,
            y=self.labels,
            batch_size=64,
            epochs=25
        )

        self.model = model

    def initial_cycle(self) -> None:
        logger.info(f"Starting Initial Data Cycle")

        prices = []
        rolling_sums = []

        for _ in range(4+self.FREQUENCY):
            while True:
                now = localtime()

                if now.tm_sec == 0:
                    prices.append(
                        float(
                            self.client.futures_symbol_ticker(
                                symbol=self.symbol
                            )["price"]
                        )
                    )

                    print(f"Initial Price -> {prices[-1]} {self.quote}")

                    if len(prices) >= 5:
                        rolling_sums.append(
                            diff(
                                prices[-5:]
                            ).sum()
                        )

                    sleep(1)

                    break

                sleep(60 - now.tm_sec)

        self.prices = prices
        self.rolling_sums = rolling_sums

    def buy(self):
        quote_balance = float(
            self.client.futures_account_balance(
                self.quote
            )['free']
        )

        if quote_balance > 20:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                type='MARKET',
                side="BUY",
                quantity=(self.amount*self.leverage) / self.prices[-1]
            )

            logger.info(f"BUY | Order Id #{order['orderId']} | Qty {order['origQty']} {config['base']}")

            self.holding = True
            self.prev_balance = quote_balance

    def sell(self):
        base_balance = float(
            self.client.futures_account_balance(
                self.base
            )["free"]
        )

        if (base_balance * self.prices[-1]) > 20:
            order = self.futures_create_order(
                symbol=self.symbol,
                type='MARKET',
                side="SELL",
                quantity=base_balance
            )

            sleep(2.5)

            quote_balance = float(
                self.client.futures_account_balance(
                    self.quote
                )['free']
            )

            logger.info(f"SELL | Order Id #{order['orderId']} | {quote_balance - self.prev_balance} {self.quote} profit")

            self.holding = False
            self.last_buy = 0

    def trade_cycle(self):
        logger.info(f"Starting Trade Cycle")

        while True:
            now = localtime()

            if now.tm_sec == 0:
                self.prices.append(
                    float(
                        self.client.futures_symbol_ticker(
                            symbol=self.symbol
                        )["price"]
                    )
                )

                print(f"Price -> {self.prices[-1]} {self.quote}")

                self.rolling_sums.append(
                    diff(
                        self.prices[-5:]
                    ).sum()
                )

                if self.holding:
                    last_sale += 1
                    
                    if self.last_sale >= 5:
                        self.sell()
                        
                    else:
                        assets = float(
                            self.client.futures_account_balance(
                                self.base
                            )["free"]
                        )
                        
                        if (assets * self.prices[-1]) > self.prev_balance:
                            self.sell()

                elif self.model.predict(array(self.rolling_sums[-self.FREQUENCY:]).reshape(1, 1, -1), verbose=0)[0][0] > self.ACCURACY:
                    self.buy()

                self.prices = self.prices[-4:]
                self.rolling_sums = self.rolling_sums[-(self.FREQUENCY-1):]

            sleep(60 - now.tm_sec)

if __name__ == "__main__":
    logging.info("Loading Config File")

    with open("config.json", "r") as config_file:
        config = loads(
            config_file.read()
        )

    logging.info("Calling Main Class")

    Krymeta(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        base=config["base"],
        quote=config["quote"],
        amount=config["amount"],
        leverage=config["leverage"],
        FREQUENCY=config["frequency"],
        ACCURACY=config["accuracy"]
    )