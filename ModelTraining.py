from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from DataPreprocessing import X, y
import pandas as pd

# vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
Xs = vectorizer.fit_transform(X['侮辱字詞特徵'])
X = X.drop(columns=['侮辱字詞特徵'])

# convert the vectorized data back to dataframe
X = pd.concat([X, pd.DataFrame(Xs.toarray())], axis=1)


print(X)

# Split the data into training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the model
model.fit(X_train, y_train)

# predict the data
y_pred = model.predict(X_test)

# calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')