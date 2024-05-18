# predict.py
import pandas as pd
import joblib
import ttkbootstrap as ttk




def changeResultValue(event=None):
    text = inputEntry.get()
    repeat_offender = repeatOffenderValue.get()
    mitigating_factor = mitigatingFactorValue.get()
    repeat_offender_while_on_parole = repeatOffenderWhileOnParoleValue.get()
    repeat_offender_record = repeatOffenderRecordValue.get()
    other_record = otherRecordValue.get()
    confession = confessionValue.get()
    attitude = attitudeValue.get()

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
    prediction = model.predict(X_full_input)

    print(f'預測罰金: {prediction}')

    # 轉換成人類可讀的格式，並且把金額變成整數(以千元為單位)，並且超過一萬元的部分轉成拘役天數或是超過三個月轉成有期徒刑月數
    if prediction > 10000:
        if prediction > 90000:
            prediction = f'{int(prediction/90000)} 月'
        else:
            prediction = f'{int(prediction/1000)} 日'
    else:# 四捨五入成千元
        prediction = f'{int(round(int(prediction)/1000,0)*1000)} 元'
        
    if inputEntry.get()=='':
        prediction = "0 元"
    print(f'預測罰金: {prediction}')
    predictResultValue.config(text=prediction)


# Load the model
model = joblib.load('insult_fine_prediction_model.pkl')
vectorizer = joblib.load('insult_fine_prediction_vectorizer.pkl')
extract_insult_features = joblib.load('insult_fine_prediction_extract_insult_features.pkl')

# Create a GUI window
root = ttk.Window(themename='cosmo')
root.title('AI 公然侮辱刑度通靈系統')
root.geometry('800x800')

titleFrame = ttk.Frame(root)
titleFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# TitleLabel
titleLabel = ttk.Label(titleFrame, text='AI 公然侮辱刑度通靈系統', font=('Helvetica', 26),anchor=ttk.W, bootstyle='primary')
titleLabel.pack(side='top', fill=ttk.X, expand=True)
subtitleLabel = ttk.Label(titleFrame, text='Insult.AI', font=('Helvetica', 16),anchor=ttk.W)
subtitleLabel.pack(side='top', fill=ttk.X, expand=True)

# 請輸入侮辱字詞，字詞請用、分隔，使用「」包住字詞
inputFrame = ttk.Frame(root)
inputFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# InputLabel
inputLabel = ttk.Label(inputFrame, text='請輸入侮辱字詞，字詞請用、分隔，使用「」包住字詞', font=('Helvetica', 12),anchor=ttk.W)
inputLabel.pack(side='top', fill=ttk.X, expand=True)

# InputEntry
inputEntry = ttk.Entry(inputFrame)
inputEntry.pack(side='top', fill=ttk.X, expand=True)

# 加重事由-累犯
repeatOffenderFrame = ttk.Frame(root)
repeatOffenderFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# RepeatOffenderLabel
repeatOffenderLabel = ttk.Label(repeatOffenderFrame, text='加重事由-累犯(無/有):\t\t\t\t\t', font=('Helvetica', 12),anchor=ttk.W)
repeatOffenderLabel.pack(side='left', fill=ttk.X, expand=True)

# RepeatOffenderSelectionRadioButtons
repeatOffenderValue = ttk.StringVar()
repeatOffenderValue.set('無')
repeatOffenderSelectionNo = ttk.Radiobutton(repeatOffenderFrame, text='無', variable=repeatOffenderValue, value='無', command=changeResultValue)
repeatOffenderSelectionNo.pack(side='right', fill=ttk.X, expand=True)
repeatOffenderSelectionYes = ttk.Radiobutton(repeatOffenderFrame, text='有', variable=repeatOffenderValue, value='有', command=changeResultValue)
repeatOffenderSelectionYes.pack(side='right', fill=ttk.X, expand=True)

# 減輕事由
mitigatingFactorFrame = ttk.Frame(root)
mitigatingFactorFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# MitigatingFactorLabel
mitigatingFactorLabel = ttk.Label(mitigatingFactorFrame, text='減輕事由(無/有):\t\t\t\t\t\t', font=('Helvetica', 12),anchor=ttk.W)
mitigatingFactorLabel.pack(side='left', fill=ttk.X, expand=True)

# MitigatingFactorSelectionRadioButtons
mitigatingFactorValue = ttk.StringVar()
mitigatingFactorValue.set('無')
mitigatingFactorSelectionNo = ttk.Radiobutton(mitigatingFactorFrame, text='無', variable=mitigatingFactorValue, value='無', command=changeResultValue)
mitigatingFactorSelectionNo.pack(side='right', fill=ttk.X, expand=True)
mitigatingFactorSelectionYes = ttk.Radiobutton(mitigatingFactorFrame, text='有', variable=mitigatingFactorValue, value='有', command=changeResultValue)
mitigatingFactorSelectionYes.pack(side='right', fill=ttk.X, expand=True)

# 行為人是否於緩刑中或假釋中再犯
repeatOffenderWhileOnParoleFrame = ttk.Frame(root)
repeatOffenderWhileOnParoleFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# RepeatOffenderWhileOnParoleLabel
repeatOffenderWhileOnParoleLabel = ttk.Label(repeatOffenderWhileOnParoleFrame, text='行為人是否於緩刑中或假釋中再犯(否/是):\t\t\t', font=('Helvetica', 12),anchor=ttk.W)
repeatOffenderWhileOnParoleLabel.pack(side='left', fill=ttk.X, expand=True)

# RepeatOffenderWhileOnParoleSelectionRadioButtons
repeatOffenderWhileOnParoleValue = ttk.StringVar()
repeatOffenderWhileOnParoleValue.set('否')
repeatOffenderWhileOnParoleSelectionNo = ttk.Radiobutton(repeatOffenderWhileOnParoleFrame, text='否', variable=repeatOffenderWhileOnParoleValue, value='否', command=changeResultValue)
repeatOffenderWhileOnParoleSelectionNo.pack(side='right', fill=ttk.X, expand=True)
repeatOffenderWhileOnParoleSelectionYes = ttk.Radiobutton(repeatOffenderWhileOnParoleFrame, text='是', variable=repeatOffenderWhileOnParoleValue, value='是', command=changeResultValue)
repeatOffenderWhileOnParoleSelectionYes.pack(side='right', fill=ttk.X, expand=True)

# 行為人有無「公然侮辱」之前案紀錄
repeatOffenderRecordFrame = ttk.Frame(root)
repeatOffenderRecordFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# RepeatOffenderRecordLabel
repeatOffenderRecordLabel = ttk.Label(repeatOffenderRecordFrame, text='行為人有無「公然侮辱」之前案紀錄(無/有):\t\t\t', font=('Helvetica', 12),anchor=ttk.W)
repeatOffenderRecordLabel.pack(side='left', fill=ttk.X, expand=True)

# RepeatOffenderRecordSelectionRadioButtons
repeatOffenderRecordValue = ttk.StringVar()
repeatOffenderRecordValue.set('無')
repeatOffenderRecordSelectionNo = ttk.Radiobutton(repeatOffenderRecordFrame, text='無', variable=repeatOffenderRecordValue, value='無', command=changeResultValue)
repeatOffenderRecordSelectionNo.pack(side='right', fill=ttk.X, expand=True)
repeatOffenderRecordSelectionYes = ttk.Radiobutton(repeatOffenderRecordFrame, text='有', variable=repeatOffenderRecordValue, value='有', command=changeResultValue)
repeatOffenderRecordSelectionYes.pack(side='right', fill=ttk.X, expand=True)

# 行為人有「公然侮辱」外之任何前案紀錄？
otherRecordFrame = ttk.Frame(root)
otherRecordFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# OtherRecordLabel
otherRecordLabel = ttk.Label(otherRecordFrame, text='行為人有「公然侮辱」外之任何前案紀錄？(無/有):\t\t\t', font=('Helvetica', 12),anchor=ttk.W)
otherRecordLabel.pack(side='left', fill=ttk.X, expand=True)

# OtherRecordSelectionRadioButtons
otherRecordValue = ttk.StringVar()
otherRecordValue.set('無')
otherRecordSelectionNo = ttk.Radiobutton(otherRecordFrame, text='無', variable=otherRecordValue, value='無', command=changeResultValue)
otherRecordSelectionNo.pack(side='right', fill=ttk.X, expand=True)
otherRecordSelectionYes = ttk.Radiobutton(otherRecordFrame, text='有', variable=otherRecordValue, value='有', command=changeResultValue)
otherRecordSelectionYes.pack(side='right', fill=ttk.X, expand=True)

# 是否坦承
confessionFrame = ttk.Frame(root)
confessionFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# ConfessionLabel
confessionLabel = ttk.Label(confessionFrame, text='是否坦承(否認/坦承/未敘明/先否認後坦承):\t\t', font=('Helvetica', 12),anchor=ttk.W)
confessionLabel.pack(side='left', fill=ttk.X, expand=True)

# ConfessionSelectionRadioButtons
confessionValue = ttk.StringVar()
confessionValue.set('否認')
confessionSelectionNo = ttk.Radiobutton(confessionFrame, text='否認', variable=confessionValue, value='否認', command=changeResultValue)
confessionSelectionNo.pack(side='right', fill=ttk.X, expand=True)
confessionSelectionYes = ttk.Radiobutton(confessionFrame, text='坦承', variable=confessionValue, value='坦承', command=changeResultValue)
confessionSelectionYes.pack(side='right', fill=ttk.X, expand=True)
confessionSelectionUnknown = ttk.Radiobutton(confessionFrame, text='未敘明', variable=confessionValue, value='未敘明', command=changeResultValue)
confessionSelectionUnknown.pack(side='right', fill=ttk.X, expand=True)
confessionSelectionBoth = ttk.Radiobutton(confessionFrame, text='先否認後坦承', variable=confessionValue, value='先否認後坦承', command=changeResultValue)
confessionSelectionBoth.pack(side='right', fill=ttk.X, expand=True)

# 犯後態度
attitudeFrame = ttk.Frame(root)
attitudeFrame.pack(fill=ttk.X, padx=20, pady=20, side='top')

# AttitudeLabel
attitudeLabel = ttk.Label(attitudeFrame, text='犯後態度(良好/尚有悔意/尚可/未敘明/不佳/無悔意):', font=('Helvetica', 12),anchor=ttk.W)
attitudeLabel.pack(side='left', fill=ttk.X, expand=True)

# AttitudeSelectionRadioButtons
attitudeValue = ttk.StringVar()
attitudeValue.set('良好')
attitudeSelectionGood = ttk.Radiobutton(attitudeFrame, text='良好', variable=attitudeValue, value='良好', command=changeResultValue)
attitudeSelectionGood.pack(side='right', fill=ttk.X, expand=True)
attitudeSelectionRemorse = ttk.Radiobutton(attitudeFrame, text='尚有悔意', variable=attitudeValue, value='尚有悔意', command=changeResultValue)
attitudeSelectionRemorse.pack(side='right', fill=ttk.X, expand=True)
attitudeSelectionOkay = ttk.Radiobutton(attitudeFrame, text='尚可', variable=attitudeValue, value='尚可', command=changeResultValue)
attitudeSelectionOkay.pack(side='right', fill=ttk.X, expand=True)
attitudeSelectionUnknown = ttk.Radiobutton(attitudeFrame, text='未敘明', variable=attitudeValue, value='未敘明', command=changeResultValue)
attitudeSelectionUnknown.pack(side='right', fill=ttk.X, expand=True)
attitudeSelectionBad = ttk.Radiobutton(attitudeFrame, text='不佳', variable=attitudeValue, value='不佳', command=changeResultValue)
attitudeSelectionBad.pack(side='right', fill=ttk.X, expand=True)
attitudeSelectionNoRemorse = ttk.Radiobutton(attitudeFrame, text='無悔意', variable=attitudeValue, value='無悔意', command=changeResultValue)
attitudeSelectionNoRemorse.pack(side='right', fill=ttk.X, expand=True)

# PredictResultFrame
predictResultFrame = ttk.Frame(root, bootstyle='primary')
predictResultFrame.pack(fill=ttk.BOTH, expand=True, padx=20, pady=20, side='top')

# PredictResultLabel
predictResultLabel = ttk.Label(predictResultFrame, text='預測罰金: ', font=('Helvetica', 16),anchor=ttk.W, bootstyle='inverse-primary')
predictResultLabel.pack(side='left', fill=ttk.X, expand=True)

# PredictResultValue
predictResultValue = ttk.Label(predictResultFrame, text='0 元', font=('Helvetica', 32),anchor=ttk.W, bootstyle='inverse-primary')
predictResultValue.pack(side='left', fill=ttk.X, expand=True)

# Bind the changeResultValue function to the inputEntry
inputEntry.bind('<KeyRelease>', changeResultValue)

# Change Result Value anytime the input changes
# 測試輸入新的案例 (侮辱字詞,加重事由-累犯,減輕事由,行為人是否於緩刑中或假釋中再犯,行為人有無「公然侮辱」之前案紀錄,行為人有「公然侮辱」外之任何前案紀錄？,是否坦承,犯後態度)
columns = ['加重事由-累犯', '減輕事由', '行為人是否於緩刑中或假釋中再犯', '行為人有無「公然侮辱」之前案紀錄',
           '行為人有「公然侮辱」外之任何前案紀錄？', '是否坦承', '犯後態度', '侮辱字詞長度'] + [f'vec_{i}' for i in range(len(vectorizer.get_feature_names_out()))]


root.mainloop()




