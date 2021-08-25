import pandas as pd
from lightgbm import LGBMClassifier
from sklearn  import preprocessing
import phonenumbers
import pickle

le = preprocessing.LabelEncoder()


df_train = pd.read_csv('data/train.csv')
df_train = df_train
df_train = df_train
df_train.dropna(inplace = True)
df_train.reset_index(drop = True, inplace = True)
df_train.pop('title')
df_train.pop('datetime_submitted')
df_train.pop('subcategory')
df_train.pop('region')
df_train.pop('city')

description = df_train.pop('description')
df_train_y = df_train.pop('is_bad')

le.fit(df_train.category)
df_train['category'] = le.transform(df_train.category)

list_word = []      # получаем список интересующих нас слов
words = "instagram inst инстаграм инста youtube ютуб facebook  face facetime  вк вконтакте vk id telegram  телегр вайбер viber вибер ватсапп вотсап whatsapp .com com сайт sms смс сообщ mail кварт номер номеру тел. телефон писать пишите написав звон мой мир одноклассники одноклассниках imessage html девять девятсот 9 91 92 93 94 95 96 97 98 90 восемь 8 семь +7 здесь @ точка  www .ru what".split()
for i in words:
    list_word.append(i)

df_train['количество слов'] = pd.Series([0] * len(df_train)) # добавили колонки для количественных признаков
df_train['количество переносов'] = pd.Series([0] * len(df_train))
df_train["кол-во '/'"] = pd.Series([0] * len(df_train))
df_train['PhoneMatch'] = pd.Series([0] * len(df_train))

for elem in list_word:         # добавили колонки для категориальных признаков вхождения слов
    df_train[elem] = pd.Series([0] * len(df_train))

for i in range(len(df_train)):
    sentense = description[i]  # считываем описание из поля 'description' индексы совпадают с i
    df_train.iloc[i, 2] = len(sentense.split())  # заполняем количесвтвенные поля
    df_train.iloc[i, 3] = len(sentense.splitlines())
    df_train.iloc[i, 4] = sentense.count('/')
    match = phonenumbers.PhoneNumberMatcher(sentense,'RU')
    if match.has_next():
        df_train.iloc[i, 5] = 1
    #  чтобы заполнить категориальные поля
    for j in range(len(list_word)):
        if sentense.lower().find(list_word[j]) != -1:
            df_train.iloc[i, 6+j] = 1
X_train = df_train
y_train = df_train_y

classifier = LGBMClassifier(random_state=40)
classifier.fit(X_train,y_train)
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
