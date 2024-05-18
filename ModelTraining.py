from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from DataPreprocessing import X, y
import pandas as pd

# vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
Xs = vectorizer.fit_transform(X['侮辱字詞特徵'])

# Drop the original text feature
X = X.drop(columns=['侮辱字詞特徵'])

# Convert the vectorized data back to dataframe with appropriate column names
Xs_df = pd.DataFrame(Xs.toarray(), columns=[f'vec_{i}' for i in range(Xs.shape[1])])

# Concatenate the original data with vectorized features
X = pd.concat([X, Xs_df], axis=1)

# Ensure all column names are strings
X.columns = X.columns.astype(str)

# Split the data into training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict the data
y_pred = model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
