from lazypredict.Supervised import LazyClassifier
from src_csv import preprocessing

x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split()

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models.sort_values(by="Accuracy", ascending=False))