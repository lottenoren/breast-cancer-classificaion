import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)[:,1]

    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

if __name__ == "__main__":
    main()
