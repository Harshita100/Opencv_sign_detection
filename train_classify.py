import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dic = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dic['data'])
labels = np.asarray(data_dic['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle=True, stratify = labels) 

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f'{score*100}% is the accuracy\n')

f = open('model.pickle', 'wb')
pickle.dump({'model':model},f)
f.close()