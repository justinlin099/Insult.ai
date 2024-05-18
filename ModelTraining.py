from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from DataPreprocessing import X, extract_insult_features, y
import pandas as pd

# Vectorize the text data using TF-IDF with n-gram
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # 使用1-2元特征
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

# 測試輸入新的案例 (侮辱字詞,加重事由-累犯,減輕事由,行為人是否於緩刑中或假釋中再犯,行為人有無「公然侮辱」之前案紀錄,行為人有「公然侮辱」外之任何前案紀錄？,是否坦承,犯後態度)
columns = ['加重事由-累犯', '減輕事由', '行為人是否於緩刑中或假釋中再犯', '行為人有無「公然侮辱」之前案紀錄',
           '行為人有「公然侮辱」外之任何前案紀錄？', '是否坦承', '犯後態度', '侮辱字詞長度'] + [f'vec_{i}' for i in range(len(vectorizer.get_feature_names_out()))]

while True:
    text = input('請輸入侮辱字詞: ')
    if text.lower() == 'exit':
        break

    repeat_offender = input('加重事由-累犯(無/有): ')
    mitigating_factor = input('減輕事由(無/有): ')
    repeat_offender_while_on_parole = input('行為人是否於緩刑中或假釋中再犯(否/是): ')
    repeat_offender_record = input('行為人有無「公然侮辱」之前案紀錄(無/有): ')
    other_record = input('行為人有「公然侮辱」外之任何前案紀錄？(無/有): ')
    confession = input('是否坦承(否認/坦承/未敘明/先否認後坦承): ')
    attitude = input('犯後態度(良好/尚有悔意/尚可/未敘明/不佳/無悔意): ')
    
    # Convert the input to the same format as the training data
    X_input = pd.DataFrame({
        '加重事由-累犯': [1 if repeat_offender == '有' else 0],
        '減輕事由': [1 if mitigating_factor == '有' else 0],
        '行為人是否於緩刑中或假釋中再犯': [1 if repeat_offender_while_on_parole == '是' else 0],
        '行為人有無「公然侮辱」之前案紀錄': [1 if repeat_offender_record == '有' else 0],
        '行為人有「公然侮辱」外之任何前案紀錄？': [1 if other_record == '有' else 0],
        '是否坦承': [2 if confession == '坦承' else 1 if confession == '先否認後坦承' else 0],
        '犯後態度': [3 if attitude == '良好' else 2 if attitude == '尚有悔意' else 1 if attitude == '尚可' else 0 if attitude == '未敘明' else -1 if attitude == '不佳' else -2],
        '侮辱字詞': [text]
    })

    # Extract features from the insult text
    X_input['侮辱字詞特徵'] = X_input['侮辱字詞'].apply(extract_insult_features)
    X_input['侮辱字詞長度'] = X_input['侮辱字詞'].apply(len)
    X_input = X_input.drop(columns=['侮辱字詞'])

    # Vectorize the text data
    X_text_vec = vectorizer.transform(X_input['侮辱字詞特徵'])
    X_text_vec_df = pd.DataFrame(X_text_vec.toarray(), columns=[f'vec_{i}' for i in range(X_text_vec.shape[1])])

    # Drop the '侮辱字詞特徵' column and concatenate the vectorized features
    X_input = X_input.drop(columns=['侮辱字詞特徵'])
    X_input = pd.concat([X_input.reset_index(drop=True), X_text_vec_df.reset_index(drop=True)], axis=1)

    # Initialize an empty DataFrame with all required columns
    X_full_input = pd.DataFrame(columns=columns)
    
    # Exclude empty or all-NA entries before concatenation
    X_full_input = pd.concat([X_full_input, X_input], axis=0, ignore_index=True, join='outer')

    # Fill missing values with 0 and infer object types
    X_full_input = X_full_input.fillna(0).infer_objects()

    # Ensure all column names are strings
    X_full_input.columns = X_full_input.columns.astype(str)
    
    # Predict the data
    prediction = best_model.predict(X_full_input)
    print(f'預測罰金: {prediction[0]} NTD')