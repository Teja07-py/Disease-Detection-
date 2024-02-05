# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



# Reading the train.csv by removing the last column since it is an empty column
#(Note: This is a dataset taken from Kaggle) 
#("https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning")

Data_set = "dataset/Training.csv"
data = pd.read_csv(Data_set).dropna(axis = 1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
	"Disease": disease_counts.index,
	"Counts": disease_counts.values
})

plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()



# Converting the desired value to a numerical value
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
 
# Model intialization
models = 
    { "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)}
 
# Cross Validation score production for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")




#Training and Testing Support Vector Machine-SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
prediction1 = svm_model.predict(X_test)
 
print(f"By SVM, accuracy on training data\: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
print(f"Accuracy on test data by SVM\: {accuracy_score(y_test, prediction1)*100}")
cf_matrix = confusion_matrix(y_test, prediction1)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM")
plt.show()
 
# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
prediction2 = nb_model.predict(X_test)
print(f"Accuracy on train data subjected to Naive Baye's Classifier\: {accuracy_score(y_train, nb_model.predict(X_train))*100}") 
print(f"Accuracy on testing data by Naive Baye's Classifier\: {accuracy_score(y_test, prediction2)*100}")

cf_matrix = confusion_matrix(y_test, prediction2)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion matrix on testing data for Naive Baye's Classifier")
plt.show()
 
# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
prediction3 = rf_model.predict(X_test)
print(f"Accuracy on Training data by RandomForestClassifier\: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
print(f"Accuracy on Testing data by RandomForestClassifier\: {accuracy_score(y_test, prediction3)*100}")
 
cf_matrix = confusion_matrix(y_test, prediction3)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion matrix for RandomForestClassifier on Test Data")
plt.show()


final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)
 
# Reading the test data
#(Note: This dataset is taken from Kaggle "https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning" )
test_data = pd.read_csv("./dataset/Testing.csv").dropna(axis=1)  
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Making predictions by taking mode of predictions made by all the classifiers
svm_pred1 = final_svm_model.predict(test_X)
nb_pred2 = final_nb_model.predict(test_X)
rf_pred3 = final_rf_model.predict(test_X)
 
final_prediction = [mode([i,j,k])[0][0] for i,j,k in zip(svm_pred1, nb_pred1, rf_pred3)] 
print(f"Accuracy on testing dataset by the combined model\: {accuracy_score(test_Y, final_prediction)*100}")
 
cf_matrix = confusion_matrix(test_Y, final_prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion matrix on testing dataset for combined model")
plt.show()



symptoms = X.columns.values

# Converting the supplied symptoms into a numerical format by building a symptom index dictionary

symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

# Defining the Function
# Here,the input is the string containing symptoms 
# and output is generated predictions by models
def predictDisease(symptoms):
	symptoms = symptoms.split(",")
	
	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
    
	input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
	return predictions

# Testing the function
print(predictDisease("X,Y,Z")) #Specify and pass symptoms as parameters

#print(predictDisease("Cough","Running Nose","High Temperature"))
#Output: Fever

