Этот репозиторий создан для ознакомления с задачей определения пола по лицу.
Изображение должно быть предобработано. Про необходимый препроцессинг можно посмотреть в introduction_to_face_recognition/face_detection

# Датасеты
Для обучения и тестирования использовались

|Dataset|Size|
|:-------:|:-----:|
|[IMDB-Wiki Faces](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), annotation [LAOFIW](http://www.robots.ox.ac.uk/~vgg/data/laofiw/)|84551|
|[UTKFaces](https://susanqq.github.io/UTKFace/)|24 021|

# GenderEstimator
GenderEstimator -- класс обработки изображения
### Вход
RGB изображения размера 112x112
### Выход
id класса, уверенность предсказания

## Пример запуска
```python
from PIL import Image

from gender_estimator import GenderEstimator

model = GenderEstimator()
img = Image.open(/path/to/img.jpg)
result = model.estimate_img(img)
````

Веса модели:

|Модель|Веса модели| 
|------|:---------:|
|IR-50| [Google Drive](https://drive.google.com/file/d/1tS5mskD0gtuhbYUtDge6yP7liMciCACM/view?usp=sharing)|

Положить в директорию weights/ рядом с gender_estimator.py

# Точность на тестовых данных

|Dataset|Size|Accuracy|
|-----|----|---------|
|IMDB-Wiki|||
|UTKFaces|||
