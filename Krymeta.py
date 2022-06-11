# Imports
# For Disabling the Warning
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For Getting The Data
from yfinance import Ticker

# For Proccesing The Data
from numpy import array, argmax
from sklearn.model_selection import train_test_split

# For Model Creation
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense

# For Ploting The Data
from matplotlib.pyplot import subplots, show

# For General Use
from datetime import timedelta

class Krymeta:
    def __init__(self, symbol):
        self.symbol = symbol

        self.dates = []

        HIGH_LOW = lambda : {
            "high": [],
            "low": []
        }

        self.prices = HIGH_LOW()

        self.getPrices()

        self.gains = HIGH_LOW()

        self.uniqueGains = HIGH_LOW()

        self.features = HIGH_LOW()
        self.labels = HIGH_LOW()

        self.xTrain = HIGH_LOW()
        self.xTest = HIGH_LOW()

        self.yTrain = HIGH_LOW()
        self.yTest = HIGH_LOW()

        self.preproccess()

        self.models = HIGH_LOW()

        self.createModel()
        
        for _ in range(len(self.yTest["high"])):
            self.predict()

        self.plot()

    def getPrices(self) -> None:
        history = Ticker(self.symbol).history(period="max").reset_index()

        self.dates = history["Date"].to_list()

        self.prices["high"] = history["High"].to_list()
        self.prices["low"] = history["Low"].to_list()

    def preproccess(self) -> None:
        for category in self.prices.keys():
            for index, price in enumerate(self.prices[category][:-1]):
                self.gains[category].append(
                    (self.prices[category][index + 1] - price) / price
                )

            self.uniqueGains[category] = list(set(self.gains[category]))

            for index in range(30, len(self.gains[category])):
                self.features[category].append(
                    self.gains[category][index-30:index]
                )

                self.labels[category].append(
                    self.uniqueGains[category].index(
                        self.gains[category][index]
                    )
                )
            
            self.features[category] = array(self.features[category]).reshape(-1, 1, 30)
            self.labels[category] = array(self.labels[category])

            self.xTrain[category], self.xTest[category], self.yTrain[category], self.yTest[category] = train_test_split(
                self.features[category],
                self.labels[category],
                test_size=0.2
            )

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
                self.xTrain[category],
                self.yTrain[category],
                epochs=10,
                batch_size=16,
                validation_data=(
                    self.xTest[category],
                    self.yTest[category]
                )
            )

            self.models[category] = model

    def predict(self) -> None:
        for category in self.prices.keys():
            self.dates.append(
                self.dates[-1] + timedelta(days=1)
            )

            prediction = self.gains[category][argmax(self.models[category].predict(
                array(self.gains[category][-30:]).reshape(-1, 1, 30),
                verbose=0
            ))]

            self.gains[category].append(prediction)

            self.prices[category].append(
                (prediction / self.gains[category][-2]) + self.gains[category][-2]
            )

    def plot(self) -> None:
        fig, ax = subplots(1, 2)

        ax[0].set_title("High")
        ax[0].plot([self.gains["high"][y] for y in self.yTest["high"]])
        ax[0].plot(self.gains["high"][-len(self.yTest["high"]):])
        ax[0].legend(["Orginal", "Prediction"])

        ax[1].set_title("Low")
        ax[1].plot([self.gains["low"][y] for y in self.yTest["low"]])
        ax[1].plot(self.gains["low"][-len(self.yTest["low"]):])
        ax[1].legend(["Orginal", "Prediction"])

        fig.savefig(f"{self.symbol}.png")

Krymeta("AAPL")

print("Done")