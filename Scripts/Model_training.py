from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle


# Load the processed data
with open(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\X_res.pkl', 'rb') as file:
    X_res = pickle.load(file)
with open(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\y_res.pkl', 'rb') as file:
    y_res = pickle.load(file)


# Define the file paths to save the data
X_train_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\X_train.pkl'
X_test_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\X_test.pkl'
y_train_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\y_train.pkl'
y_test_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\y_test.pkl'


X_train, X_test, y_train, y_test = train_test_split( X_res, y_res, test_size=0.4, random_state=42)

# Instantiate the model
model = RandomForestClassifier(random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Save X_train
with open(X_train_path, 'wb') as file:
    pickle.dump(X_train, file)

# Save X_test
with open(X_test_path, 'wb') as file:
    pickle.dump(X_test, file)

# Save y_train
with open(y_train_path, 'wb') as file:
    pickle.dump(y_train, file)

# Save y_test
with open(y_test_path, 'wb') as file:
    pickle.dump(y_test, file)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))

# Save the trained model
with open('../models/trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)





