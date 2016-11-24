import pandas as pd

red_wine = pd.read_csv('winequality-red.csv', sep=";")
white_wine = pd.read_csv('winequality-white.csv', sep=";")

red_wine.head()
white_wine.head()

red_wine.describe()
white_wine.describe()
