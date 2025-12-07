import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.loss_history = []
        self.mean = None
        self.std_deviation = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, predictions, labels):
        return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    def standardize(self, features, fit=True):
        features = np.asarray(features, dtype=np.float64)
        if fit:
            self.mean = np.mean(features, axis=0)
            self.std_deviation = np.std(features, axis=0)
        return (features - self.mean) / self.std_deviation

    def train(self, features, labels, scale=True):
        features = np.array(features)
        labels = np.array(labels)

        if scale:
            features = self.standardize(features, fit=True)

        m, n = features.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.iterations):
            z = np.dot(features, self.weights) + self.bias
            predictions = self.sigmoid(z)

            weight_gradient = (1/m) * np.dot(features.T, (predictions - labels))
            bias_gradient = (1/m) * np.sum(predictions - labels)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            current_loss = self.loss(predictions, labels)
            self.loss_history.append(current_loss)

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.iterations}, Loss: {current_loss}")

    def predict(self, features, scale=True):
        features = np.array(features)
        if scale and self.mean is not None:
            features = self.standardize(features, fit=False)

        z = np.dot(features, self.weights) + self.bias
        probabilities = self.sigmoid(z)

        predictions = (probabilities >= 0.5).astype(int)
        return predictions

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.grid(True)
        plt.show()

    def evaluate_accuracy(self, features, labels):
        if features.shape[0] != labels.shape[0]:
            print("Cannot evaluate accuracy, features size must match labels size.")
            return
        predictions = self.predict(features, scale=True)
        accuracy_array = predictions == labels
        return np.mean(accuracy_array)
    
data = 'data/Loan_approval_data_2025.csv'
df = pd.read_csv(data)
df.drop(['customer_id'], axis=1, inplace=True)
categorical = [var for var in df.columns if df[var].dtype=='O']

df = pd.get_dummies(df, columns=categorical, drop_first=True)
features = df.drop(['loan_status'], axis=1)
labels = df['loan_status']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 50)

model = LogisticRegressionClassifier(learning_rate=0.01, iterations=5000)
model.train(features_train, labels_train, scale=True)

accuracy = model.evaluate_accuracy(features_test, labels_test)
print(f"\nModel Accuracy: {accuracy}")
print(f"Final Loss: {model.loss_history[-1]}")

model.plot_loss()