Этот репозиторий создан для ознакомления с задачей определения этнической принадлежности по лицу.
Изображение должно быть предобработано. Про необходимый препроцессинг можно посмотреть в introduction_to_face_recognition/face_detection

# Датасеты
Для обучения и тестирования использовались

|Dataset|Size|
|:-------:|:-----:|
|[LAOFIW](http://www.robots.ox.ac.uk/~vgg/data/laofiw/)|11 216|
|[UTKFaces](https://susanqq.github.io/UTKFace/)|22 313|

# GenderEstimator
EthnicityEstimator -- класс обработки изображения
### Вход
RGB изображения размера 112x112
### Выход
id класса, название класса 

## Пример запуска
```python
from PIL import Image

from ethnicity_estimator import EthnicityEstimator

model = EthnicityEstimator()
img = Image.open(/path/to/img.jpg)
result = model.estimate_img(img)
````

Веса модели:

|Модель|Веса модели| 
|------|:---------:|
|IR-50| [Google Drive](https://drive.google.com/file/d/1tS5mskD0gtuhbYUtDge6yP7liMciCACM/view?usp=sharing)|

Положить в директорию weights/ рядом с ethnicity_estimator.py

# Точность на тестовых данных
## LAOFIW

|Class|Size|Precision|
|-----|----|---------|
|Black (0)|||
|White (1)|||
|Asian (2)|||
|Indian (3)|||

## UTKFaces

|Class|Size|Precision|
|-----|----|---------|
|Black (0)|||
|White (1)|||
|Asian (2)|||
|Indian (3)|||
