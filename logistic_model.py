from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_logistic(model, X_test, y_test):
    pred = model.predict(X_test)

    print("\n===== Logistic Regression =====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))