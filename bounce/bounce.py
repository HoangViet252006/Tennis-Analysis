from idlelib.replace import replace

from lazypredict.Supervised import LazyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sktime.datatypes._panel._convert import from_2d_array_to_nested

class bounce():
    def load_model(self):
        pass  # Placeholder for model loading logic

    def train(self, window_size):
        data = pd.read_csv("df_bounce.csv")

        # Add lag features for 'x', 'y', and 'V'
        for i in range(window_size):
            data[f'lagX_{i}'] = data['x'].shift(-i)
            data[f'lagY_{i}'] = data['y'].shift(-i)
            data[f'lagV_{i}'] = data['V'].shift(-i)

        # Drop original columns
        data.drop(['x', 'y', 'V'], axis=1, inplace=True)

        # Prepare nested data
        Xs = data[[f'lagX_{i}' for i in range(20)]]
        Ys = data[[f'lagY_{i}' for i in range(20)]]
        Vs = data[[f'lagV_{i}' for i in range(20)]]

        Xs = from_2d_array_to_nested(Xs.to_numpy())
        Ys = from_2d_array_to_nested(Ys.to_numpy())
        Vs = from_2d_array_to_nested(Vs.to_numpy())

        # Combine features
        X = pd.concat([Xs, Ys, Vs], axis=1)


if __name__ == '__main__':
    # bounce_predict = bounce()
    # bounce_predict.train()

    data = pd.read_csv("df_bounce.csv")
    data.drop('Order', axis=1, inplace=True)

    # Add lag features for 'x', 'y', and 'V'
    window_size =20
    train_ratio = 0.8
    for i in range(window_size):
        data[f'lagX_{i}'] = data['x'].shift(-i)
        data[f'lagY_{i}'] = data['y'].shift(-i)
        data[f'lagV_{i}'] = data['V'].shift(-i)

    # Drop original columns
    data.drop(['x', 'y', 'V'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)

    # Prepare nested data
    Xs = data[[f'lagX_{i}' for i in range(window_size)]]
    Ys = data[[f'lagY_{i}' for i in range(window_size)]]
    Vs = data[[f'lagV_{i}' for i in range(window_size)]]

    Xs = from_2d_array_to_nested(Xs.to_numpy())
    Ys = from_2d_array_to_nested(Ys.to_numpy())
    Vs = from_2d_array_to_nested(Vs.to_numpy())

    # Combine features
    X = pd.concat([Xs, Ys, Vs], axis=1)

    x = data.drop("bounce", axis=1)
    y = data["bounce"]
    num_samples = len(y)
    x_train = x[:int(num_samples * train_ratio)]
    y_train = y[:int(num_samples * train_ratio)]
    x_test = x[int(num_samples * train_ratio):]
    y_test = y[int(num_samples * train_ratio):]

    model = SVC()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    for i, j in zip(y_predict, y_test):
        if i == 1:
            print(f"Predicted value: {i}. Actual value: {j}")

    # print(f"acc: {accuracy_score(y_test, y_predict)}")
    print(classification_report(y_test, y_predict, zero_division=0))

