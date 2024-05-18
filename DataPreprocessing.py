import pandas as pd
import spacy

# Load the data
data = pd.read_csv('公然侮辱罪_地方法院判決.csv')

# Convert the data to a dataframe
df = pd.DataFrame(data)

# Display the first 5 rows of the dataframe
#print(df.head())

# convert punishment to the same unit(NTD) ; 1 day = 1000 NTD
# example 有期徒刑4月 120000
# example 拘役20日 20000
# example 新臺幣8000元 8000
def convert_punishment_to_NTD(punisment):
    if '月' in punisment:
        return int(punisment.replace('有期徒刑','').replace('月',''))*1000*30
    elif '日' in punisment:
        return int(punisment.replace('拘役','').replace('日',''))*1000
    elif '元' in punisment:
        return int(punisment.replace('新臺幣','').replace('新台幣','').replace('元',''))
    elif '無罪' in punisment:
        return 0
    else:
        return 0
    
# convert punishment to NTD
df['罰金'] = df['處罰方式'].apply(convert_punishment_to_NTD)

#print(df['處罰方式'])

# convert other variables to numerical data
# 判決字號,處罰方式,侮辱字詞,加重事由-累犯,減輕事由,行為人是否於緩刑中或假釋中再犯,行為人有無「公然侮辱」之前案紀錄,行為人有「公然侮辱」外之任何前案紀錄？,是否坦承,犯後態度
df['加重事由-累犯'] = df['加重事由-累犯'].map({'無': 0, '有': 1})
df['減輕事由'] = df['減輕事由'].map({'無': 0, '有': 1})
df['行為人是否於緩刑中或假釋中再犯'] = df['行為人是否於緩刑中或假釋中再犯'].map({'否': 0, '是': 1})
df['行為人有無「公然侮辱」之前案紀錄'] = df['行為人有無「公然侮辱」之前案紀錄'].map({'無': 0, '有': 1})
df['行為人有「公然侮辱」外之任何前案紀錄？'] = df['行為人有「公然侮辱」外之任何前案紀錄？'].map({'無': 0, '有': 1})
df['是否坦承'] = df['是否坦承'].map({'否認': 0, '坦承': 2, '未敘明': 2, '先否認後坦承': 1})
df['犯後態度'] = df['犯後態度'].map({'良好': 3,'尚有悔意': 2, '尚可': 1, '未敘明': 0, '不佳': -1, '無悔意': -2})



# initialize spacy
nlp = spacy.load('zh_core_web_sm')

def extract_insult_features(text):
    doc = nlp(text)
    return ' '.join([token.text for token in doc])

df['侮辱字詞特徵'] = df['侮辱字詞'].apply(extract_insult_features)
df['侮辱字詞長度'] = df['侮辱字詞'].apply(len)


# Display the first 5 rows of the dataframe
#print(df)

X = df.drop(columns=['判決字號', '處罰方式', '侮辱字詞', '罰金'])
y = df['罰金']