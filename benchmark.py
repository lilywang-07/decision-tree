import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('datasets/mushroom_fixed.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# same 60/20/20 split
n = len(X)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)


'''
X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[val_end:],   y[val_end:]'''

# train
t0 = time.time()
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
t1 = time.time()

# evaluate
preds = clf.predict(X_test)
print(f"scikit-learn decision tree: {accuracy_score(y_test, preds)*100:.1f}% in {(t1-t0)*1000:.4f} ms")

# Random Forest
t0 = time.time()
rf = RandomForestClassifier(criterion='entropy', random_state=42)
rf.fit(X_train, y_train)
t1 = time.time()
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest: {rf_acc*100:.1f}% in {(t1-t0)*1000:.2f}ms")

