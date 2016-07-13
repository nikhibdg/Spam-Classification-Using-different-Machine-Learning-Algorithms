import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import pickle




def main():
	init()
	
	
	# get the dataset 
	print colored("dataset Location?")
	print colored('files might get deleted if they are incompatible with utf8', 'yellow')
	ans = sys.stdin.readline()

	path = ans.strip('\n')
	if path.endswith(' '):
		path = path.rstrip(' ')


	print colored("Reorganizing folders, into two classes")
	reorganize_dataset(path)


	print '\n\n'

	# do the main test
	main_test(path)

def reorganize_dataset(path):


	folders = glob.glob(path + '/*')
	if len(folders) == 2:
		return
	

def main_test(path = None):
	dir_path = path or 'dataset'

	remove_incompatible_files(dir_path)

	print '\n\n'

	# load data
	print colored('Loading files into memory', 'green', attrs=['bold'])
	files = sklearn.datasets.load_files(dir_path)
	print len(files)
	print len(files.data)
	# refine all emails
	print colored('Refining all files', 'green', attrs=['bold'])
	util.refine_all_emails(files.data)

	# calculate the BOW representation
	#vectorizer = CountVectorizer(min_df=1)
	print colored('Calculating BOW', 'green', attrs=['bold'])
	word_counts = util.bagOfWords(files.data)

	# TFIDF
	print colored('Calculating TFIDF', 'green', attrs=['bold'])
	tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
	X = tf_transformer.fit_transform(word_counts)
	#print len(X)

	print '\n\n'

	# create classifier
	clf = sklearn.naive_bayes.MultinomialNB()
	clf1 = sklearn.svm.LinearSVC()
	n_neighbors = 11
	weights = 'uniform'
	weights = 'distance'
	clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	clf3 = DecisionTreeClassifier(random_state=0)
	clf4 = RandomForestClassifier(n_estimators=10)
	eclf = VotingClassifier(estimators=[('nb', clf), ('svm', clf1), ('knn', clf2), ('dt', clf3), ('rt', clf4)], voting='hard')
	#eclf = VotingClassifier(estimators=[('nb', clf), ('svm', clf1), ('knn', clf2), ('dt', clf3), ('rt', clf4)], voting='soft')
	
	# test the classifier
	print '\n\n'
	print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
	test_classifier(X, files.target, clf,test_size=0.4, y_names=files.target_names, confusion=False)
	test_classifier1(X, files.target, clf1,test_size=0.4, y_names=files.target_names, confusion=True)
	test_classifier2(X, files.target, clf2,test_size=0.4, y_names=files.target_names, confusion=True)
	test_classifier3(X, files.target, clf3,test_size=0.4, y_names=files.target_names, confusion=True)
	test_classifier4(X, files.target, clf4,test_size=0.4, y_names=files.target_names, confusion=True)
	test_classifier5(X, files.target, eclf,test_size=0.4, y_names=files.target_names, confusion=True)



def remove_incompatible_files(dir_path):
	# find incompatible files
	print colored('Finding files incompatible with utf8: ', 'green', attrs=['bold'])
	incompatible_files = util.find_incompatible_files(dir_path)
	print colored(len(incompatible_files), 'yellow'), 'files found'

	# delete them
	if(len(incompatible_files) > 0):
		print colored('Deleting incompatible files', 'red', attrs=['bold'])
		util.delete_incompatible_files(incompatible_files)

def test_classifier(X, y, clf,test_size=0.4, y_names=None, confusion=False):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)
	print 'Naive Bayes'

	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	
	accuracy = accuracy_score(y_predicted,y_test)
	
	print accuracy	
	
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)

		precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_predicted)
		plt.plot(recall, precision)

		plt.show()
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(0,1):
    		fpr[i], tpr[i], _ = roc_curve(y_test[i], y_predicted[i])
    		roc_auc[i] = auc(fpr[i], tpr[i])

		fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



		plt.figure()
		plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()

	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)
	
	#pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=2)


def test_classifier4(X, y, clf4,test_size=0.4, y_names=None, confusion=True):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	clf4.fit(X_train, y_train)
	y_predicted = clf4.predict(X_test)
	
	accuracy = accuracy_score(y_predicted,y_test)
	print accuracy	
	print 'Random Forest'
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

def test_classifier3(X, y, clf3,test_size=0.4, y_names=None, confusion=True):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	clf3.fit(X_train, y_train)
	y_predicted = clf3.predict(X_test)
	
	
	accuracy = accuracy_score(y_predicted,y_test)
	print accuracy	
	print 'Decision Tree'
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)


def test_classifier1(X, y, clf1,test_size=0.4, y_names=None, confusion=True):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)
	
	clf1.fit(X_train, y_train)
	y_predicted = clf1.predict(X_test)
	accuracy = accuracy_score(y_predicted,y_test)
	print accuracy
	print 'SVM'
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

def test_classifier2(X, y, clf2,test_size=0.4, y_names=None, confusion=True):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)
	print 'KNN'
	clf2.fit(X_train, y_train)
	y_predicted = clf2.predict(X_test)
	accuracy = accuracy_score(y_predicted,y_test)
	print accuracy
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

def test_classifier5(X, y, eclf,test_size=0.4, y_names=None, confusion=True):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	eclf.fit(X_train, y_train)
	y_predicted = eclf.predict(X_test)
	
	accuracy = accuracy_score(y_predicted,y_test)
	print accuracy	
	print 'Hard Voting'
	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

if __name__ == '__main__':
	main()
