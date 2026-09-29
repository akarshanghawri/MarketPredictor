from pandas import read_csv
from train_baseline import get_features_and_labels
from sklearn.metrics import accuracy_score
import xgboost as xgb 

series = read_csv('data/processed/featured_data.csv')
dates = series['DATE'].values
X,y = get_features_and_labels(series)

n_train = 584
step_size = 120
n_folds = 5

accuracies = []
naive_accuracies = []

prev_test_end = None
for fold in range(n_folds):
    train_end = n_train + (fold * step_size)
    test_start = train_end
    test_end = test_start + step_size

    if prev_test_end is not None:
        assert test_start >= prev_test_end, "test folds overlap"
    prev_test_end = test_end

    if test_end > len(X):
        break

    assert dates[train_end - 1] < dates[test_start], "train/test overlap or out of order"

    X_train = X.iloc[0:train_end]
    X_test = X.iloc[test_start:test_end]

    y_train = y.iloc[0:train_end]
    y_test = y.iloc[test_start:test_end]

    model = xgb.XGBClassifier()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracies.append(accuracy)

    majority_class = y_train.mode()[0]
    naive_acc = (y_test == majority_class).mean()
    naive_accuracies.append(naive_acc)

    print('Fold %d: train=%d (%s to %s), test=%d (%s to %s), accuracy=%.2f%%' % (
    fold + 1,
    len(X_train), dates[0], dates[train_end - 1],
    len(X_test), dates[test_start], dates[test_end - 1],
    accuracy * 100
    ))

average_model_acc = sum(accuracies) / len(accuracies)
average_naive_acc = sum(naive_accuracies) / len(naive_accuracies)

print("Average accuracy :", average_model_acc)
print("Average naive accuracy :", average_naive_acc)

