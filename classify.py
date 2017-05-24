import zipfile
import random
import glob
import os
import sklearn
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


def preprocess(data):
    """
    Function which removes
    not important characters

    """

    final_message =\
            data.strip("Subject").strip(":").strip("\n").strip("\r").strip("re").strip("-")

    return final_message


def get_features(messages):

    """
    Function which converts the
    messages into feature or vectorize form

    """
    vectorizer = CountVectorizer(min_df=1)
    X_count_vec = vectorizer.fit_transform(messages)
    transformer = TfidfTransformer(smooth_idf=False)
    X = transformer.fit_transform(X_count_vec)

    return X


def read_data(curr_dir):
    """
    Function which reads the
    two directories and preprocess
    and creates messages and labels list

    """

    messages_list = []
    temp_list = []
    messages = []
    labels = []

    # Reading Ham data
    ham_folder = glob.glob(curr_dir + "/dataset/ham" + "/*.txt")
    for file_item in ham_folder:
        with open(file_item, "r") as fp:
            ham_message = preprocess(fp.readline())
            messages_list.append({ham_message: 1})

    # Reading Spam data
    spam_folder = glob.glob(curr_dir + "/dataset/spam" + "/*.txt")
    for file_item in spam_folder:
        with open(file_item, "r") as fp:
            spam_message = preprocess(fp.readline())
            temp_list.append({spam_message: 0})
            messages_list.append({spam_message: 0})

    random_spam_mssgs = random.sample(temp_list, 1000)
    messages_list.extend(random_spam_mssgs)

    random.shuffle(messages_list)
    for item in messages_list:
        for key, value in item.items():
            messages.append(key)
            labels.append(value)

    return messages, labels


def test_classifier_nb(X_train, X_test, y_train, y_test, nb_clf):
    """
    Function which tests
    Naive Bayes classifier
    outputs the accuracy and confusion matrix

    """

    print("Naive Bayes\n")
    nb_clf.fit(X_train, y_train)
    y_predicted = nb_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("Naive Bayes Accuracy:", accuracy)
    print("Naive Bayes Classification Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("Naive Bayes Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier_svm(X_train, X_test, y_train, y_test, svm_clf):
    """
    Function which tests
    SVM classifier
    outputs the accuracy and confusion matrix

    """

    print("\n\nSupport Vector Machine\n")
    svm_clf.fit(X_train, y_train)
    y_predicted = svm_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("SVM Accuracy:", accuracy)
    print("SVM Classification Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("SVM Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier_knn(X_train, X_test, y_train, y_test, knn_clf):
    """
    Function which tests
    KNN classifier
    outputs the accuracy and confusion matrix

    """

    print("\n\nKNN\n")
    knn_clf.fit(X_train, y_train)
    y_predicted = knn_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("KNN Accuracy:", accuracy)
    print("KNN Classification Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("KNN Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier_dtree(X_train, X_test, y_train, y_test, dtree_clf):
    """
    Function which tests
    Decision Tree classifier
    outputs the accuracy and confusion matrix

    """

    print("\n\nDecision Tree\n")
    dtree_clf.fit(X_train, y_train)
    y_predicted = dtree_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("Decision Tree Accuracy:", accuracy)
    print("Decision Classification Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("Decision Tree Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier_rforest(X_train, X_test, y_train, y_test, rforest_clf):
    """
    Function which tests
    Random Forest classifier
    outputs the accuracy and confusion matrix

    """

    print("\n\nRandom Forest\n")
    rforest_clf.fit(X_train, y_train)
    y_predicted = rforest_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("Random Forest Accuracy:", accuracy)
    print("Random Forest Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("Random Forest Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier_voting(X_train, X_test, y_train, y_test, vote_clf):
    """
    Function which
    does hard voting of all classifiers
    outputs the accuracy and confusion matrix

    """

    print("\n\nVoting Classifier\n")
    vote_clf.fit(X_train, y_train)
    y_predicted = vote_clf.predict(X_test)

    accuracy = accuracy_score(y_predicted, y_test)
    print("Voting Classifier Accuracy:", accuracy)
    print("Voting Classifier Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
                                                target_names=None))

    print("Voting Classifier Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


if __name__ == "__main__":

    n_neighbors = 11
    weights = 'distance'
    curr_dir = dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.isdir("dataset"):
        print("Directory Exists, no need of unzipping dataset")
    else:
        print("Zipping the datasets into two folders Spam and Ham")
        zip_pointer = zipfile.ZipFile("dataset.zip", 'r')
        zip_pointer.extractall(curr_dir)
        zip_pointer.close()

    messages, labels = read_data(curr_dir)
    features = get_features(messages)
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.33,
                                                        random_state=42)

    nb_clf = sklearn.naive_bayes.MultinomialNB()
    svm_clf = sklearn.svm.LinearSVC()
    knn_clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors,
                                                     weights=weights)
    dtree_clf = DecisionTreeClassifier(random_state=0)
    rforest_clf = RandomForestClassifier(n_estimators=10)
    vote_clf = VotingClassifier(estimators=[('nb', nb_clf), ('svm', svm_clf),
                                            ('knn', knn_clf),
                                            ('dt', dtree_clf),
                                            ('rt', rforest_clf)],
                                voting='hard')

    # testing the classifier

    test_classifier_nb(X_train, X_test, y_train, y_test,  nb_clf)
    test_classifier_svm(X_train, X_test, y_train, y_test,  svm_clf)
    test_classifier_knn(X_train, X_test, y_train, y_test,  knn_clf)
    test_classifier_dtree(X_train, X_test, y_train, y_test,  dtree_clf)
    test_classifier_rforest(X_train, X_test, y_train, y_test, rforest_clf)
    test_classifier_voting(X_train, X_test, y_train, y_test, vote_clf)
