import pickle
import pandas as pd
from sklearn import preprocessing
import phonenumbers

le = preprocessing.LabelEncoder()

def task1(df):
    with open('data/model.pkl', 'rb') as f:
        clf = pickle.load(f)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.pop('title')
    df.pop('datetime_submitted')
    df.pop('subcategory')
    df.pop('region')
    df.pop('city')

    description2 = df.pop('description')

    le.fit(df.category)
    df['category'] = le.transform(df.category)

    list_word = []
    words = 'instagram inst инстаграм инста youtube ютуб facebook  face facetime  вк вконтакте vk id telegram  телегр вайбер viber вибер ватсапп вотсап whatsapp .com com сайт sms смс сообщ mail кварт номер номеру тел. телефон писать пишите написав звон мой мир одноклассники одноклассниках imessage html девять девятсот 9 91 92 93 94 95 96 97 98 90 восемь 8 семь +7 здесь @ точка  www .ru what'.split()
    for i in words:
        list_word.append(i)

    df['количество слов'] = pd.Series([0] * len(df))
    df['количество переносов'] = pd.Series([0] * len(df))
    df["кол-во '/'"] = pd.Series([0] * len(df))
    df['PhoneMatch'] = pd.Series([0] * len(df))

    for elem in list_word:
        df[elem] = pd.Series([0] * len(df))

    for i in range(len(df)):
        sentense = description2[i]
        df.iloc[i, 2] = len(sentense.split())
        df.iloc[i, 3] = len(sentense.splitlines())
        df.iloc[i, 4] = sentense.count('/')
        match = phonenumbers.PhoneNumberMatcher(sentense, 'RU')
        if match.has_next():
            df.iloc[i, 5] = 1
        for j in range(len(list_word)):
            if sentense.lower().find(list_word[j]) != -1:
                df.iloc[i, 6 + j] = 1

    y_pred = clf.predict_proba(df)
    return y_pred[:, 1]


def task2(df):
    starts = list()
    finals = list()
    for text in df['description']:
        k = 0
        for match in phonenumbers.PhoneNumberMatcher(text, "RU"):
            k += 1
            if k == 1:
                start = match.start
                final = match.end
            else:
                start = None
                final = None
                break
        if k == 0:
            start = None
            final = None
        starts.append(start)
        finals.append(final)
    return starts, finals
