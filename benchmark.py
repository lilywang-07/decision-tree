import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('datasets/mushroom_fixed.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# same 60/20/20 split
n = len(X)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[val_end:],   y[val_end:]

# train
t0 = time.time()
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
t1 = time.time()

# evaluate
preds = clf.predict(X_test)
print(f"scikit-learn accuracy: {accuracy_score(y_test, preds)*100:.1f}%")
print(f"scikit-learn training time: {(t1-t0)*1000:.4f} ms")