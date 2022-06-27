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
from tensorflow.keras.optimizers import SGD, Adam

# For Ploting The Data
from matplotlib.pyplot import clf, show, subplots

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
        self.yTrain = HIGH_LOW()
        
        self.xTest = HIGH_LOW()
        self.yTest = HIGH_LOW()

        self.preproccess()

        self.models = HIGH_LOW()

        self.createModel()
        
        for _ in range(100):
            self.predict()

        self.plot()

    def getPrices(self) -> None:
        history = Ticker(self.symbol).history(period="max").reset_index()

        self.dates = history["Date"].to_list()

        self.prices["high"] = history["High"].to_list()
        self.prices["low"] = history["Low"].to_list()

    def preproccess(self) -> None:
        for category in self.prices.keys():
            for index in range(1, len(self.prices[category])):
                self.gains[category].append(
                    round(
                        self.prices[category][index] - self.prices[category][index-1]
                    )
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
                test_size=0.2,
                shuffle=False
            )

    def createModel(self) -> None:
        for category in self.prices.keys():
            model = Sequential()

            model.add(LSTM(
                units=128,
                input_shape=(
                    self.features[category].shape[1],
                    self.features[category].shape[2]
                ),
                return_sequences=True
            ))

            model.add(Bidirectional(LSTM(
                units=64,
                return_sequences=True
            )))

            model.add(Bidirectional(LSTM(
                units=32,
                return_sequences=True
            )))

            model.add(Bidirectional(LSTM(
                units=16,
                return_sequences=False
            )))

            model.add(Dense(
                len(self.labels[category]),
                activation="softmax"
            ))

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=Adam(
                    learning_rate=0.0001,
                    decay=0.00001
                ),
                metrics=["accuracy"]
            )

            for l in range(2):
                print("\n", "Test" if l == 0 else "Train", "\n")

                model.fit(
                    self.xTrain[category] if l == 0 else self.features[category],
                    self.yTrain[category] if l == 0 else self.labels[category],
                    epochs=25,
                    batch_size=16,
                    validation_data=(
                        self.xTest[category],
                        self.yTest[category]
                    ) if l == 0 else None
                )

            self.models[category] = model

    def predict(self) -> None:
        for category in self.prices.keys():
            self.dates.append(
                self.dates[-1] + timedelta(days=1)
            )

            prediction = self.gains[category][argmax(self.models[category].predict(
                array(self.prices[category][-30:]).reshape(-1, 1, 30),
                verbose=0
            ))]

            self.gains[category].append(prediction)

            self.prices[category].append(
                prediction + self.prices[category][-1]
            )

    def plot(self) -> None:
        fig, ax = subplots(2, 1)

        ax[0].set_title("Predicted Prices")
        ax[0].plot(self.prices["high"][-100:])
        ax[0].plot(self.prices["low"][-100:])
        ax[0].legend(["High", "Low"])

        ax[1].set_title("All Prices")
        ax[1].plot(self.prices["high"])
        ax[1].plot(self.prices["low"])
        ax[1].legend(["High", "Low"])

        fig.savefig(f"{self.symbol}.png") 

        show()

Krymeta("msft")

print("Done")