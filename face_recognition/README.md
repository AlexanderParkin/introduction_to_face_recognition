Этот репозиторий создан для ознакомления с задачей распознавания лиц.
Изображение должно быть предобработано. Про необходимый препроцессинг можно посмотреть в introduction_to_face_recognition/face_detection

# Источник
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

# FaceRecognizer
FaceRecognizer -- класс обработки изображения
### Вход
RGB изображения размера 112x112
### Выход
дескриптор (numpy массив) размера 1x512

## Пример запуска
```python
from PIL import Image

from face_recognition import FaceRecognizer

model = FaceRecognizer()
img = Image.open(/path/to/img.jpg)
result = model.get_descriptor(img)
````

Веса модели:

|Модель|Веса модели| 
|------|:---------:|
|IR-50| [Google Drive](https://drive.google.com/file/d/1Qh_z-OnIuJt-BNQw4OMpYx_i-_rzV-WS/view?usp=sharing)|

Положить в директорию weights/ рядом с face_recognition.py
