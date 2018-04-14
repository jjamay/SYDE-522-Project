from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


def test_gbc(x_tr, x_ts, y_tr, y_ts):
    gbt = GradientBoostingClassifier(max_features="log2")
    gbt.fit(x_tr, y_tr)

    p = gbt.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy
