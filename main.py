import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


# Загрузим данные из файла, создадим dataframe
pd.set_option('display.max_columns', None)
data = pd.read_csv('Data_Set.csv', index_col=0)

# Уберем сталь номер 50 из данных
data = data[data['Steel Grade'] != 50]

# Общее количество данных
print('Общее количество данных', data.shape[0])

# Зависимоть сопротивления деформации от обжатия для разных марок сталей
Thick_Red = pd.DataFrame(data['Thick Red.'])
Thick_Red['sigma'] = data['sigma']
Thick_Red['Steel Grade'] = data['Steel Grade']
Thick_Red = Thick_Red.sort_values(by='Thick Red.')
for i in Thick_Red['Steel Grade'].unique():
    Thick_Red_1 = Thick_Red[Thick_Red['Steel Grade'] == i]
    plt.scatter(Thick_Red_1['Thick Red.'], Thick_Red_1['sigma'])
    plt.ylabel(f'Сопротивление деформации стали {i}, МПа')
    plt.xlabel('Обжатие, %')
    plt.title("Зависисмость сопротивления деформации от обжатия")
    plt.grid(True)
    plt.show()

# Общаяя зависимость сопротивления деформации от обжатия для всего набора данных
plt.scatter(Thick_Red['Thick Red.'], Thick_Red['sigma'])
plt.ylabel(f'Сопротивление деформации, МПа')
plt.xlabel('Обжатие, %')
plt.title("Зависисмость сопротивления деформации от обжатия")
plt.grid(True)
plt.show()

columns = ['h0',
           'h1',
           'Thick Red.',
           'Width',
           'Length',
           'Speed',
           'Diam. Top',
           'Diam. Bot',
           'time',
           'temperature',
           'speed_deform.',
           'Pass',
           'Steel Grade']

# Входные параметры и target
X = data[columns]
Y = data['sigma']

# Разобьем данные на обучающую, тестовую и валидационную выборку
x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
print('-' * 30)
print('Количество данных для обучения', x_train.shape[0])
print('Количество данных для валидации', x_val.shape[0])
print('Количество данных для теста', x_test.shape[0])
print('-' * 30)

# Создадим модель многослойной нейронной сети
model = keras.Sequential([Dense(units=256, input_shape=(13,), activation='relu'),
                          Dense(128, activation='relu'),
                          Dense(64, activation='relu'),
                          Dense(32, activation='relu'),
                          Dense(16, activation='relu'),
                          Dense(8, activation='relu'),
                          Dense(1, activation='linear')
                          ])

# Выберем функцию потерь и оптимизатор
model.compile(loss='mean_absolute_percentage_error', optimizer=keras.optimizers.Adam())
# Начинаем обучение
history = model.fit(x_train, y_train, batch_size=500, epochs=500, verbose=1,
                    validation_data=(x_val, y_val), shuffle=True)

# Построим график ошибки и валидации
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Cредний абсолютный процент ошибок')
plt.xlabel('Эпоха обучения')
plt.legend(['На обучающем наборе', 'На проверочной выборке'], loc='upper right')
plt.title("Точность нейронной сети")
plt.grid(True)
plt.show()

# Вывод структуры НС
print(model.summary())

# Проверка на тестовых данных
result = pd.DataFrame({'sigma_ист': y_test})
result['sigma_расч'] = model.predict(x_test)
result['Ошибка_%_НС'] = (abs(result['sigma_ист'] - result['sigma_расч']) * 100) / result['sigma_ист']
print('-' * 30)
print('Средняя ошибка НС в %:', result['Ошибка_%_НС'].sum() / result.shape[0])
print('-' * 30)

# График реальных и предсказанных значений НС
plt.scatter(x_test['Thick Red.'], y_test)
plt.scatter(x_test['Thick Red.'], model.predict(x_test))
plt.ylabel(f'Сопротивление деформации, МПа')
plt.xlabel('Обжатие, %')
plt.legend(['Данные со стана', 'Предсказания НС'], loc='upper left')
plt.title("Зависисмость сопротивления деформации от обжатия")
plt.grid(True)
plt.show()

# График для наглядности! Только предсказанные значения.
plt.scatter(x_test['Thick Red.'], model.predict(x_test), color='g')
plt.ylabel(f'Сопротивление деформации, МПа')
plt.xlabel('Обжатие, %')
plt.legend(['Предсказания НС'], loc='upper left')
plt.title("Зависисмость сопротивления деформации от обжатия")
plt.grid(True)
plt.show()

# Применим Градиентный бустинг
clf = CatBoostRegressor(
    loss_function='RMSE',
    learning_rate=0.01,
    depth=10,
    task_type='GPU')

clf.fit(x_train, y_train, verbose=0, eval_set=(x_val, y_val))
print('Модель CatBoost установлена : ' + str(clf.is_fitted()))
print('Параметры модели CatBoost:', clf.get_params())

# График важности входных данных
print('-' * 30)
print(clf.get_feature_importance(prettified=True))
plt.figure(figsize=(12, 4))
plt.barh(clf.get_feature_importance(prettified=True)['Feature Id'],
         clf.get_feature_importance(prettified=True)['Importances'])
plt.ylabel('Признаки')
plt.xlabel('Влияние признаков на результат')
plt.show()

# Проверка на тестовых данных
print('-' * 30)
result['CatBoost'] = clf.predict(x_test)
result['Ошибка_%_CatBoost'] = (abs(result['sigma_ист'] - result['CatBoost']) * 100) / result['sigma_ист']
print('Точность модели CatBoost:', clf.score(x_test, y_test))
print('Средняя ошибка в %:', result['Ошибка_%_CatBoost'].sum() / result.shape[0])
print(result.head())
print('-' * 30)

# График реальных и предсказанных значений CatBoost на тестовой выборке
plt.scatter(x_test['Thick Red.'], y_test)
plt.scatter(x_test['Thick Red.'], clf.predict(x_test))
plt.ylabel(f'Сопротивление деформации, МПа')
plt.xlabel('Обжатие, %')
plt.legend(['Данные со стана', 'Предсказания CatBoost'], loc='upper left')
plt.title("Зависисмость сопротивления деформации от обжатия")
plt.grid(True)
plt.show()

# Проверим модель на данных стали номер 50
df = pd.read_csv('sigma_gleeble.csv')
Thick_Red_1 = df[df['Steel Grade'] == 50]
df_50 = pd.DataFrame({'sigma': df['sigma']})
df_50['CatBoost'] = clf.predict(df[columns])
df_50['Ошибка_%_CatBoost_50'] = (abs(df_50['sigma'] - df_50['CatBoost']) * 100) / df_50['sigma']
print('-' * 30)
print('Точность модели CatBoost:', clf.score(df[columns], df['sigma']))
print('Средняя ошибка в %:', df_50['Ошибка_%_CatBoost_50'].sum() / df_50.shape[0])
print('-' * 30)

# График реальных и предсказанных значений CatBoost для стали номер 50
plt.scatter(df['Thick Red.'], df['sigma'])
plt.scatter(df['Thick Red.'], clf.predict(df[columns]))
plt.ylabel(f'Сопротивление деформации для стали номер 50, МПа')
plt.xlabel('Обжатие, %')
plt.legend(['Данные стана', 'Предсказания CatBoost'], loc='upper left')
plt.title("Зависисмость сопротивления деформации от обжатия для стали 50")
plt.grid(True)
plt.show()
