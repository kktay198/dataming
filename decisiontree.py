import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score
# Building a decision tree model

train_df=pd.read_csv('preprocessed_train.csv')
test_df=pd.read_csv('preprocessed_test.csv')

x_train = train_df.iloc[:, :-1].values  # All columns as features except for the last column
y_train = train_df['income-per-year_encoded'].values   # Income-per-year_encoded as target

x_test = test_df.iloc[:, :-1].values  # All columns as features except for the last column
y_test = test_df['income-per-year_encoded']  # Only the last column (target/labels


treeClassifier = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
treeClassifier.fit(x_train, y_train)

# Make predictions on the test set

start_time = time.time()

y_pred = treeClassifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Stop the timer after training
end_time = time.time()

print("Training Time is", (end_time-start_time))