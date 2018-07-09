# Acoustic_event_solution
STC ML school problem solving

## Описание решения
Используется [VGGish модель от Google Audioset](https://github.com/tensorflow/models/tree/master/research/audioset). Из нее получаются векторы признаков (средний вектор вложений). Далее применяется SVM.

### Подготовительные мероприятия

0) В консоле в корневой папке проекта запустить pip install -r requirements.txt
1) Скачать [tf версию предобученной модели](https://storage.googleapis.com/audioset/vggish_model.ckpt)
2) Положить vggish_model.ckpt в папку models
3) В папку data распоковать содержимое конкурсных данных

### Закрытая задача 
1) В консоли в корневой папке проекта набрать python train.py (можно пропустить, в папке model уже лежит обученная модель)
2) В консоли в корневой папке проекта набрать python eval.py (решение закрытой задачи)
3) Решение будет лежать в data/result.txt

### Открытая задача
Насколько понятно, в октрытой задаче могут быть метки классов, отсуствующие в обучающих данных.
В решении открытой задачи предполагает использовать [google ontology](https://research.google.com/audioset/index.html).
#### Наивное решение
Основано на VGGish, подмножестве audioset, KNN классификаторе с одним соседом. Возвращает лейблы из audioset.
1) Скачать мини-датасет [отсюда](https://drive.google.com/file/d/1hx9KmGs_KnEvq8XycSkVdcnWKxutTqLY/view?usp=sharing)
2) Распаковать и положить папку mini_audioset в data/ontology
3) В консоли в корневой папке проекта набрать python open_task_train.py (можно пропустить, в папке model уже лежит обученная модель)
4) В консоли в корневой папке проекта набрать python open_task_eval.py (решение открытой задачи)
5) Решение будет лежать в data/open_task_result.txt