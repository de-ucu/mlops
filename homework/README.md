## Звіт про виконання домашньої роботи №1

### Модель

В якості моделі було обрано архітектуру, ResNet50V2 з keras, без вагів з imagenet. Вона була натренована на датасеті з котами і собаками з завдання.

В голові моделі був використаний один класб з сігмоїдою в якості активації та бінарною кросентропією в якості функції втрат.

Для трекінгу прогресу тренування був обраний neptune.ai

Модель тренувалася 50 епох, але найкращі результати були досягнуті після 30 епох - validation accuracy 0.874, validation loss - 0.3

[Ноутбук](./services/model-preparation/notebooks/02-model_2_classes.ipynb)

### Датасет

В якості датасету був використаний датасет з завдання. Він був розділений на тренувальний та валідаційний датасети в пропорції 80 / 20.

Зображення були переформатовані до розміру `224х224х3` з додаванням падінгів за необхідності.

Далі зображення були запаковані в `tfrecords` для прискорення читання з диску.

На етапі тренування були додані такі аугментації - випадкове відображення по горизонталі або вертикалі, випадковий поворот та випадкова зміна контрасту.

```python
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomContrast(0.1),
])
```

[Ноутбук](./services/model-preparation/notebooks/01-tfrecords_2_classes.ipynb)

### DVC

Версіонування датасетів було виконано за допомогою DVC. Щоб не ускладнювати процес, в якості сховища була використана локальна директорія:

```
dvc init --subdir
dvc remote add -d remote ../dvc_storage`:
```
Потім були додані, і за необхідності оновлювались, директорії `data` і `model`:

```bash
dvc add data
dvc add model
dvc commit
dvc push
```

### Розширення датасету

Третім класом було обрано датасет з фото коней [звідси](https://www.kaggle.com/alessiocorrado99/animals10) - 2623 фото. Дані було перерозподілено випадковим чином, і знову запаковано в `tfrecords`

[Ноутбук](./services/model-preparation/notebooks/03-tfrecords_3_classes.ipynb)

### Інференс

Для забезпечення інференсу було обрано бібліотеку triton inference server, яка візьме на себе розподіл навантаження проміж відеокартами, та за необхідності буде збирати запити в батчі. Triton був доданий в docker-compose.

Щоб організувати передачу запитів від клієнта в triton, в docker-compose ще була додана служба api - [код](./services/api/src/server.py). Там же виконується і предобробка.

Модель була конвертована до формату `onnx` для оптимізації, під час якої було видалено непотрібні шари [ноутбук](./services/model-preparation/notebooks/06-convert_to_onnx.ipynb)

Щоб протестувати класифікатор, достатньо виконати команду `curl -F image=@/path/to/image.jpeg http://localhost:5000/predict`

### AutoML

Для автоматичного подбору моделі і параметрів був використаний autokeras. Загалом було перевірено 9 комбінацій архітектур і гіперпараметрів, але жодного разу валідаційна точність не перевищила 0.49

[Ноутбук](./services/model-preparation/notebooks/04-model_3_classes_automl.ipynb)

