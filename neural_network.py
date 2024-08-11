import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Neural:
    def __init__(self, data) -> None:
        self.data = data
        self.weights = None
        self.b = None

    def preprocess_data(self):
        X = self.data.data
        Y = self.data.target
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train).T
        x_test = scaler.transform(x_test).T
        # x_test = x_test.T
        # x_train = x_train.T

        y_train = (5 == y_train).astype(int).reshape(1, -1)
        y_test = (5 == y_test).astype(int).reshape(1, -1)

        self.data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

    def lost_func(self, a: np.ndarray, y: np.ndarray):
        return np.sum(y * np.log10(a) + (1.0 - y) * np.log10(1.0 - a)) / y.shape[1]

    def sigmod(self, y: np.ndarray):
        return 1 / (1 + np.exp(-y))

    def neural(self):
        x = self.data["x_train"]
        y = self.data["y_train"]

        b = 1
        m = x.shape[1]
        weights = np.ones((x.shape[0], 1))
        min_lost = -1000

        for _ in range(10000):
            z = weights.T.dot(x) + b
            a = self.sigmod(z)
            lf = self.lost_func(a, y)
            print("lost func:", lf)
            if lf > min_lost:
                min_lost = lf
                self.weights = weights
                self.b = b
            dz = a - y
            dw = x.dot(dz.T) / m
            db = np.sum(dz) / m
            rate = 0.8
            weights -= rate * dw
            db -= rate * db

        print("minimum lost:", min_lost)

    def threshold(self, x):
        return (x >= 0.6).astype(int)

    def predict(self, x):
        z = self.weights.T.dot(x) + self.b
        a = self.sigmod(z)
        return a

    def plot_precision_recall_vs_threshold(self):
        x_test = self.data["x_test"]
        y_test = self.data["y_test"].reshape(
            -1,
        )
        y_prob = self.predict(x_test).reshape(
            -1,
        )

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

        plt.figure()
        plt.plot(thresholds, precision[:-1], "b--", label="Precision")
        plt.plot(thresholds, recall[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Precision/Recall")
        plt.title("Precision and Recall vs Threshold")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


data = load_digits()
nn = Neural(data)

nn.preprocess_data()

nn.neural()

y_pred = nn.predict(nn.data["x_test"]).reshape(
    -1,
)
y_pred = nn.threshold(y_pred)
y_test = nn.data["y_test"].reshape(
    -1,
)

print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("accuracy:", accuracy_score(y_test, y_pred))

nn.plot_precision_recall_vs_threshold()
