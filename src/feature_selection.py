from sklearn.feature_selection import SelectKBest, f_classif


def select_features(X_train, y_train, X_test, k=100):
    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector
