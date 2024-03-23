import pandas as pd

df=pd.read_csv('D:\\Git-HUB\\artifact\\train.csv')


x_train=df[:,:-1]

print(x_train)