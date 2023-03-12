# Krymeta
Krymeta is a program designed to predict cryptocurrency prices using machine learning. Its name derives from the Greek word "chrimata," meaning money. Originally, it was intended to be a tool for stock price prediction, but the cost of obtaining historical data and APIs for stock trading proved prohibitive. As a result, the project pivoted to trading cryptocurrencies, as they are more easily accessible and data is available for free.

The program uses a machine learning model generated with Keras to predict whether a cryptocurrency can be bought at the current price and sold for a profit within the next 5 minutes. The program utilizes Binance as a historical data and trading API service.

# Features
The program offers the following features:

1. Predicts cryptocurrency prices using machine learning
2. Uses Binance API for historical data and trading
3. Predicts whether a cryptocurrency can be bought now and sold for a profit within the next 5 minutes
4. Generates constant profits

# Requirements
The following are required to run the program:

1. Python 3.6+
2. A free Binance account.
3. Spot balance of with more than 20 usd.

# Installation
To install the program, follow these steps:

1. Clone the repository to your local machine.
2. Create a new virtual environment: python -m venv venv
3. Activate the environment: ./venv/Scripts/activate
4. Install the required packages: pip install -r requirements.txt
5. Obtain a Binance API key and secret from https://www.binance.com/en/usercenter/settings/api-management
6. Replace the values in the config.py file with your API key and secret
7. Run the program with python Krymeta.py

# Support
If you encounter any issues or have any questions, please contact the developer at sh3ldr0id@gmail.com.

# Note
Things to note before using Krymeta.

1. Get rich quick isn't a goal of krymeta. It's supposed to generate small but constant profits.
2. I'm not responsible for any kind of losses caused by the program.
3. The best trading pair to use on this program according to me could be BTCBUSD as binance has a 0 fee policy for it..