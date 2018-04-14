from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def test_rfc(x_tr, x_ts, y_tr, y_ts):
    rft = RandomForestClassifier()
    rft.fit(x_tr, y_tr)

    p = rft.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy
