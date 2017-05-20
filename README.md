# Spam-Classification-Using-different-Machine-Learning-Algorithms

This code unzips the dataset into Spam and Ham(Normal Emails)
Further it converts the Spam and Ham messages into proper vector forms using
Sklearn count Vectorizer and  TFIDF Vectorizer

After that the dataset is spilt in 0.33 and 0.77
The 0.77 part of Dataset is trained on various classifiers, this trained models
are further tested on the left out 0.33 part of dataset

The accuracy, classificaion reports and confusion matrices of each classifiers
are printed

Steps to Run
1. Install the dependencies using requirements.txt
   i.e pip3 install requirements.txt
2. Run classify.py
