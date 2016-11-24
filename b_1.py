import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

red_wine = pd.read_csv('winequality-red.csv',sep=";")
white_wine = pd.read_csv('winequality-white.csv',sep=";")

red_wine.drop('quality',axis=1,inplace=True)
white_wine.drop('quality',axis=1,inplace=True)

both_wine = pd.concat([red_wine,white_wine])
both_wine.to_pickle('both_wine.pkl')