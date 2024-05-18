from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error
from DataPreprocessing import X, y, extract_insult_features
import pandas as pd
import joblib


# vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
Xs = vectorizer.fit_transform(X['侮辱字詞特徵'])
joblib.dump(vectorizer, 'insult_fine_prediction_vectorizer.pkl')

# Drop the original text feature
X = X.drop(columns=['侮辱字詞特徵'])

# Convert the vectorized data back to dataframe with appropriate column names
Xs_df = pd.DataFrame(Xs.toarray(), columns=[f'vec_{i}' for i in range(Xs.shape[1])])

# Concatenate the original data with vectorized features
X = pd.concat([X, Xs_df], axis=1)

# Ensure all column names are strings
X.columns = X.columns.astype(str)

# Split the data into training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=20)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=20)

# Fit the model
model.fit(X_train, y_train)

# Predict the data
y_pred = model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model

joblib.dump(model, 'insult_fine_prediction_model.pkl')



# # 測試輸入新的案例 (侮辱字詞,加重事由-累犯,減輕事由,行為人是否於緩刑中或假釋中再犯,行為人有無「公然侮辱」之前案紀錄,行為人有「公然侮辱」外之任何前案紀錄？,是否坦承,犯後態度)
# columns = ['加重事由-累犯', '減輕事由', '行為人是否於緩刑中或假釋中再犯', '行為人有無「公然侮辱」之前案紀錄',
#            '行為人有「公然侮辱」外之任何前案紀錄？', '是否坦承', '犯後態度', '侮辱字詞長度'] + [f'vec_{i}' for i in range(len(vectorizer.get_feature_names_out()))]

# while True:
#     text = input('請輸入侮辱字詞: ')
#     if text.lower() == 'exit':
#         break

#     repeat_offender = input('加重事由-累犯(無/有): ')
#     mitigating_factor = input('減輕事由(無/有): ')
#     repeat_offender_while_on_parole = input('行為人是否於緩刑中或假釋中再犯(否/是): ')
#     repeat_offender_record = input('行為人有無「公然侮辱」之前案紀錄(無/有): ')
#     other_record = input('行為人有「公然侮辱」外之任何前案紀錄？(無/有): ')
#     confession = input('是否坦承(否認/坦承/未敘明/先否認後坦承): ')
#     attitude = input('犯後態度(良好/尚有悔意/尚可/未敘明/不佳/無悔意): ')
    
#     # Convert the input to the same format as the training data
#     X_input = pd.DataFrame({
#         '加重事由-累犯': [1 if repeat_offender == '有' else 0],
#         '減輕事由': [1 if mitigating_factor == '有' else 0],
#         '行為人是否於緩刑中或假釋中再犯': [1 if repeat_offender_while_on_parole == '是' else 0],
#         '行為人有無「公然侮辱」之前案紀錄': [1 if repeat_offender_record == '有' else 0],
#         '行為人有「公然侮辱」外之任何前案紀錄？': [1 if other_record == '有' else 0],
#         '是否坦承': [2 if confession == '坦承' else 1 if confession == '先否認後坦承' else 0],
#         '犯後態度': [3 if attitude == '良好' else 2 if attitude == '尚有悔意' else 1 if attitude == '尚可' else 0 if attitude == '未敘明' else -1 if attitude == '不佳' else -2],
#         '侮辱字詞': [text]
#     })

#     # Extract features from the insult text
#     X_input['侮辱字詞特徵'] = X_input['侮辱字詞'].apply(extract_insult_features)
#     X_input['侮辱字詞長度'] = X_input['侮辱字詞'].apply(len)
#     X_input = X_input.drop(columns=['侮辱字詞'])

#     # Vectorize the text data
#     X_text_vec = vectorizer.transform(X_input['侮辱字詞特徵'])
#     X_text_vec_df = pd.DataFrame(X_text_vec.toarray(), columns=[f'vec_{i}' for i in range(X_text_vec.shape[1])])

#     # Drop the '侮辱字詞特徵' column and concatenate the vectorized features
#     X_input = X_input.drop(columns=['侮辱字詞特徵'])
#     X_input = pd.concat([X_input.reset_index(drop=True), X_text_vec_df.reset_index(drop=True)], axis=1)

#     # Initialize an empty DataFrame with all required columns
#     X_full_input = pd.DataFrame(columns=columns)
    
#     # Exclude empty or all-NA entries before concatenation
#     X_full_input = pd.concat([X_full_input, X_input], axis=0, ignore_index=True, join='outer')

#     # Fill missing values with 0 and infer object types
#     X_full_input = X_full_input.fillna(0).infer_objects()

#     # Ensure all column names are strings
#     X_full_input.columns = X_full_input.columns.astype(str)
    
#     # Predict the data
#     prediction = model.predict(X_full_input)

#     # 轉換成人類可讀的格式，並且把金額變成整數(以千元為單位)，並且超過一萬元的部分轉成拘役天數或是超過三個月轉成有期徒刑月數
#     if prediction > 10000:
#         if prediction > 90000:
#             prediction = f'{int(prediction/90000)} 月'
#         else:
#             prediction = f'{int(prediction/10000)*10} 日'
#     else:# 四捨五入成千元
#         prediction = f'{int(round(int(prediction)/1000,0)*1000)} 元'
        

#     print(f'預測罰金: {prediction}')