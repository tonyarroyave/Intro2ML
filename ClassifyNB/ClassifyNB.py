def classify(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    fit = clf.fit(features_train, labels_train)
    return fit
