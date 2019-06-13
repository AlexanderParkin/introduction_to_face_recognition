Этот репозиторий создан для ознакомления с задачей определения возраста по лицу.
Изображение должно быть предобработано. Про необходимый препроцессинг можно посмотреть в introduction_to_face_recognition/face_detection

# Датасеты
Для обучения и тестирования использовались

|Dataset|Size|
|:-------:|:-----:|
|[IMDB-Wiki Faces](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), annotation [LAOFIW](http://www.robots.ox.ac.uk/~vgg/data/laofiw/)|59 979|
|[MegaAge](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/)|41 544|
|[MegaAge Asian](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/)|43 277|
|[UTKFaces](https://susanqq.github.io/UTKFace/)|24 021|

# AgeEstimator
AgeEstimator -- класс обработки изображения
### Вход
RGB изображения размера 112x112
### Выход
возраст

## Пример запуска
```python
from PIL import Image

from age_estimator import AgeEstimator

model = AgeEstimator()
img = Image.open(/path/to/img.jpg)
result = model.estimate_img(img)
````

Веса модели:

|Модель|Веса модели| 
|------|:---------:|
|IR-50| [Google Drive](https://drive.google.com/file/d/1f1MAb8tYCrtY1bswp1iK7NaNTtDsI3ZW/view?usp=sharing)|

Положить в директорию weights/ рядом с age_estimator.py

# Точность на тестовых данных

|Dataset|Size|MAE|
|-----|----|---------|
|IMDB-Wiki|||
|MegaAge|||
|MegaAge Asian|||
|UTKFaces|||
