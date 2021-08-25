# 1. Описание

Решается две задачи:
 1. Определить есть ли в объявлении контактная информация 
 2. Найти положение контактной информации в описании объявлении


## 1.1 Описание данных
Для обучения и инференса есть следующие поля:
* `title` - заголовок,
* `description` - описание,
* `subcategory` - подкатегория,
* `category` - категория,
* `price` - цена,
* `region` - регион,
* `city` - город,
* `datetime_submitted` - дата размещения.

Таргет первой задачи: `is_bad`. Для второй разметка не предоставляется.

Есть два датасета: `train.csv` и `val.csv`. 
В датасетах могут встречаться некорректные метки.

`train.csv` содержит больше данных, однако разметка в нём менее точная.

В `val.csv` существенно меньше данных, но более точная разметка.

`val.csv` находится в папке `./data`. 
`train.csv` можно качать скриптом `./data/get_train_data.sh` или перейдя по 
[ссылке](https://drive.google.com/file/d/1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL/view?usp=sharing) 

## 1.2 Задача 1
В первой задаче необходимо оценить вероятность наличия в объявлении контактной информации. 
Результатом работы модели является `pd.DataFrame` с колонками:
* `index`: `int`, положение записи в файле;
* `prediction`: `float` от 0 до 1.

## 1.3 Задача 2

Во второй задаче необходимо предсказать начало и конец контактной информации в описании (`description`) объявления. 
Например:
* для строки `Звоните на +7-888-888-88-88, в объявлении некорректный номер`: (11, 26),
* для строки `Звоните на +7-888aaaaa888aaaa88a88, в объявлении некорректный номер`: (11, 33),
* для строки `мой tg: @ivanicki_i на звонки не отвечаю`: (8, 18),
* для строки `мой tg: ivanicki_i на звонки не отвечаю`: (8, 17),
* если в описании объявления (поле `description`) контактов нет, то (None, None)
* если в описании объявления (поле `description`) более одного контакта (`Звоните не 89990000000 или на 89991111111`), то (None, None).

Результатом работы модели является `pd.DataFrame` с колонками:
* `index`: `int`, положение записи в файле;
* `start`: `int` or `None`, начало маски контакта;
* `end`: `int` or `None`, конец маски контакта.\
(`start` < `end`)
  

