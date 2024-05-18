from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from DataPreprocessing import X, y
import pandas as pd

# Vectorize the text data using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
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
model = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict the data
y_pred = best_model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
