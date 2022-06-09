# Imports
# For Disabling the Warning
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For Getting The Data
from yfinance import Ticker

# For Proccesing The Data
from numpy import array, argmax

# For Model Creation
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense

# For Ploting The Data
from matplotlib.pyplot import title, plot, legend, show

class Krymeta:
    def __init__(self, symbol):
        self.symbol = symbol

        self.prices = {
            "high": [],
            "low": []
        }

        self.getPrices()

        self.gains = {
            "high": [],
            "low": []
        }

        self.features = {
            "high": [],
            "low": []
        }
        self.labels = {
            "high": [],
            "low": []
        }

        self.preproccess()

        self.models = {
            "high": None,
            "low": None
        }

        self.createModel()
        
        for _ in range(100):
            self.predict()

        title(self.symbol)

        plot(self.gains["high"])
        plot(self.gains["high"][:-100])

        # plot(self.prices["low"])
        # plot(self.prices["low"][:-100])

        legend(["High Prediction", "High"])

        show()

    def getPrices(self) -> None:
        history = Ticker(self.symbol).history(period="max")

        self.prices["high"] = history["High"].to_list()
        self.prices["low"] = history["Low"].to_list()

    def preproccess(self) -> None:
        for category in self.prices.keys():
            for index, price in enumerate(self.prices[category][:-1]):
                self.gains[category].append(
                    (self.prices[category][index + 1] - price) / price
                )

            for index in range(60, len(self.gains[category])):
                self.features[category].append(
                    self.gains[category][index-60:index]
                )

                self.labels[category].append(
                    self.gains[category][index]
                )
            
            self.features[category] = array(self.features[category]).reshape(-1, 1, 60)
            self.labels[category] = array(self.labels[category])

    def createModel(self) -> None:
        for category in self.prices.keys():
            model = Sequential()

            model.add(LSTM(
                units=64,
                input_shape=(
                    self.features[category].shape[1],
                    self.features[category].shape[2]
                ),
                return_sequences=True
            ))

            model.add(Bidirectional(LSTM(
                units=32,
                return_sequences=False
            )))

            model.add(Dense(
                len(self.labels[category]),
                activation="softmax"
            ))

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
            )

            model.fit(
                self.features[category],
                self.labels[category],
                epochs=2,
                batch_size=8
            )

            self.models[category] = model

    def predict(self):
        for category in self.prices.keys():
            prediction = argmax(self.models[category].predict(
                array(self.gains[category][-60:]).reshape(-1, 1, 60)
            ))

            print(prediction)

            self.gains[category].append(prediction)

            self.prices[category].append(
                (prediction / self.gains[category][-1]) + self.gains[category][-1]
            )

Krymeta("AAPL")