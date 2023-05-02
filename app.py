from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys
from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
data = pd.read_csv("Crop_recommendation.csv")
df_boston = data
df_boston.columns = df_boston.columns
df_boston.head()

'''Detection'''

# IQR
Q1 = np.percentile(df_boston['rainfall'], 25, interpolation='midpoint')
Q3 = np.percentile(df_boston['rainfall'], 75, interpolation='midpoint')

IQR = Q3-Q1
print("Old Shape", df_boston.shape)
# upper bound
upper = np.where(df_boston['rainfall'] >= (Q3+1.5*IQR))

# lower bound
lower = np.where(df_boston['rainfall'] <= (Q1-1.5*IQR))

# removing the outliers
df_boston.drop(upper[0], inplace=True)
df_boston.drop(lower[0], inplace=True)
print("New Shape:", df_boston.shape)

x = data.drop('label', axis=1)
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, shuffle=True, random_state=0)
model = lgb.LGBMClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(
    accuracy_score(y_test, y_pred)))
# @app.route('/train', methods=['POST'])
# def train():
#     # Load data from POST request
#     data = request.get_json()

#     # Convert data to pandas DataFrame
#     df = pd.DataFrame.from_dict(data)

#     # Split data into features and target
#     X = df.drop('target', axis=1)
#     y = df['target']

#     # Train decision tree model
#     clf = DecisionTreeClassifier()
#     clf.fit(X, y)

#     return 'Model trained successfully!'


@app.route('/predict', methods=['POST'])
def predict():
    # Load data from POST request
    data = request.get_json()

    # Convert data to pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    print('called\n\n')
    print(df)
    # Load trained model from file
    clf = lgb.LGBMClassifier()
    clf.fit(x_train, y_train)
    # clf.fit(df)
    # clf = clf.load('model.pkl')

    # Make predictions
    preds = clf.predict(df)
    print(preds.tolist())
    # Return predictions as JSON
    return jsonify(predictions=preds.tolist())
    # return jsonify("response")


if __name__ == '__main__':
    app.run(debug=True)
