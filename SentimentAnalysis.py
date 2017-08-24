#Importing Required Libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

#Importing test_data and train_data
print "Importing data: "
if __name__=='__main__':
	train=pd.read_csv('labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
	test=pd.read_csv('testData.tsv',header=0,delimiter='\t',quoting=3)
	unlabeled_train=pd.read_csv('unlabeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

#Word2vec Implementation
print "Generating word vectors..."

clean_train_reviews=[]
for i in xrange(len(train["review"])):
	clean_train_reviews.append(" ".join(KaggleWord2VecUtility.reviewto_wordlist(train["review"][i],True)))

unlabeled_clean_train_reviews = []
for review in unlabeled_train['review']:
    unlabeled_clean_train_reviews.append( " ".join( KaggleWord2VecUtility.reviewto_wordlist( review )))

#Creating the Bag of Words
print "Creating vector..."
vector=TfidfVectorizer(analyzer="word",max_features=50000,sublinear_tf=True,stop_words = 'english',ngram_range=(1, 2), use_idf=1,smooth_idf=1,strip_accents='unicode',min_df=3)

#Tokenizing the vectors
print "Tokenizing the vector..." 
vector=vector.fit(clean_train_reviews+unlabeled_clean_train_reviews)
train_data=vector.transform(clean_train_reviews)


y=train["sentiment"]

#splitting train data for testing purposes
print "Splitting training data for testing..."
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2,random_state=42)


showdown=False
op=True

#Showdown(removed Gaussian as performed poorly)
if showdown:
	print "Classifier showdown..."
	classifiers=[
				RandomForestClassifier(n_estimators=150),
				MultinomialNB(alpha=0.0001),	
				SGDClassifier(loss='modified_huber',warm_start="True"),
				LogisticRegression(penalty="l2",C=1)
				]
	count=0
	for clf in classifiers:
		count+=1
		print "Training...",count
		clf.fit(X_train,y_train)
		print "Testing...",count		
		y_pred=clf.predict(X_test)
		print "Result...",count,":",accuracy_score(y_test,y_pred)
if op:
	print "Training classifier..."
	clf=LogisticRegression(penalty="l2",C=1) #performing better than others
	clf.fit(train_data,y)

	print "Training complete."

	clean_test_reviews=[]
	print "Creating test data..."
	for i in xrange(len(test["review"])):
		clean_test_reviews.append(" ".join(KaggleWord2VecUtility.reviewto_wordlist(test["review"][i],True)))
	test_data=vector.transform(clean_test_reviews)

	print "Testing..."
	y_pred=clf.predict_proba(test_data)[:,1]
	print "Testing complete."
	print "Preparing submission file..."
	submission=pd.DataFrame(data={"id":test['id'],"sentiment":y_pred})
	submission.to_csv('submission.csv',quoting=3,index=False)
