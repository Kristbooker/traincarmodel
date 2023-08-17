import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
from sklearn.ensemble import VotingClassifier  


data_train=pickle.load(open(r'hogvectors_train.pkl','rb'))
data_test=pickle.load(open(r'hogvectors_test.pkl','rb'))
X_data_train=[]
Y_data_train=[]
X_data_test=[]
Y_data_test=[]

X_data_train = [entry[:8100] for entry in data_train]
Y_data_train = [entry[8100] for entry in data_train]
label_encoder_train = LabelEncoder()
Y_numeric_train = label_encoder_train.fit_transform(Y_data_train)


X_data_test = [entry[:8100] for entry in data_test]
Y_data_test = [entry[8100] for entry in data_test]
label_encoder_test = LabelEncoder()
Y_numeric_test = label_encoder_test.fit_transform(Y_data_test)


# X_train, X_test, y_train, y_test = train_test_split(X_data_train, Y_numeric_train, test_size=0.2)
model_tree = DecisionTreeClassifier(random_state=42)
model_xgb = xgb.XGBClassifier(objective="multi;softmax",num_class=len(Y_numeric_train),random_state=42)


# Decision tree and XGBoost together using Voting Classifier
ensemble_model = VotingClassifier(estimators=[('decision_tree', model_tree), ('xgb', model_xgb)], voting='hard',weights=[1,4])
ensemble_model.fit(X_data_train, Y_numeric_train)
y_pred = ensemble_model.predict(X_data_test)
accuracy = accuracy_score(Y_numeric_test, y_pred)
print("Ensemble Accuracy:", accuracy)
confusion_mat=confusion_matrix(Y_numeric_test,y_pred)
print(confusion_mat)

write_path_model="model.pkl"
pickle.dump(ensemble_model,open(write_path_model,"wb"))
print("done")
# Only Decision Tree #
# model_tree = model_tree.fit(X_data_train,Y_numeric_train)
# y_pred = model_tree.predict(X_data_test)
# print("Decision Tree Accuracy:",accuracy_score(Y_numeric_test, y_pred))
# confusion_mat=confusion_matrix(Y_numeric_test,y_pred)
# print(confusion_mat)

# Only Xgboost #
# model_xgb = model_xgb.fit(X_data_train,Y_numeric_train)
# y_pred = model_xgb.predict(X_data_test)
# print("Accuracy:",accuracy_score(Y_numeric_test, y_pred))