import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.mean = None
        self.std_dev = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, predictions, labels):
        return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    def normalize(self, features, fit=True):
        features = np.asarray(features, dtype=np.float64)
        if fit:
            self.mean = np.mean(features, axis=0)
            self.std_dev = np.std(features, axis=0)
        return (features - self.mean) / self.std_dev

    def train(self, features, labels, scale=True):
        features = np.array(features)
        labels = np.array(labels)

        if scale:
            features = self.normalize(features, fit=True)

        training_size, num_features = features.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        loss_history = []

        for i in range(self.iterations):
            z = np.dot(features, self.weights) + self.bias
            predictions = self.sigmoid(z)

            weight_grad = (1/training_size) * np.dot(features.T, (predictions - labels))
            bias_grad = (1/training_size) * np.sum(predictions - labels)

            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad

            current_loss = self.loss(predictions, labels)
            loss_history.append(current_loss)

            if (i + 1) % 500 == 0:
                print(f"Iteration {i+1}/{self.iterations}, Loss: {current_loss}")
        return loss_history

    def predict(self, features, scale=True):
        features = np.array(features)
        if scale:
            features = self.normalize(features, fit=False)

        z = np.dot(features, self.weights) + self.bias
        probability = self.sigmoid(z)

        prediction = 1 if probability >= 0.5 else 0
        return prediction

    def test_model(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)
        if features.shape[0] != labels.shape[0]:
            raise ValueError
        accuracy_array = []
        for i in range(features.shape[0]):
            if labels[i] == self.predict(features[i], scale=True):
                accuracy_array.append(True)
            else:
                accuracy_array.append(False)
        return np.mean(accuracy_array)

def plot_loss(loss_history):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    data = 'data/Loan_approval_data_2025.csv'
    df = pd.read_csv(data)
    df.drop(['customer_id'], axis=1, inplace=True)
    categorical = [var for var in df.columns if df[var].dtype=='O']

    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    features = df.drop(['loan_status'], axis=1)
    labels = df['loan_status']
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

    model = Classifier(0.01, 2000)
    loss_history = model.train(features_train, labels_train, scale=True)
    plot_loss(loss_history)

    try:
        test_accuracy = model.test_model(features_test, labels_test)
    except ValueError as e:
        print("Unable to compute test accuracy: feature sample size must match label sample size.")
        test_accuracy = None
    
    print(f"Model Test Accuracy: {test_accuracy}")
    print(f"Final Loss: {loss_history[-1]}")