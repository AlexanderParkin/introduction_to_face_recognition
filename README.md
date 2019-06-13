# Введение в распознавание лиц

В данном репозитории собраны модели, решающие различные задачи из области распознавания лиц.

Пайплайн работы с лицами:
1. Детекция лиц на изображении
2. Нормализация лиц (поворот и приведение к одному размеру)
3. Запуск модели для определенной задачи, получение результа.

# Задачи распознавания лиц

|Task name| Input | Output | Path |
|---------|-------|--------|------|
|Face detection| full size image|face bbox, landmark5 points|`face_detection/mtcnn/`|
|Face alignment|img, bbox, landmark5 points|aligned 112x112 image|`face_detection/mtcnn/arcface_warping.py`|
|Face recognition|aligned 112x112 img|face descriptor 1x512||
|Age recognition|aligned 112x112 img|float age|`age_recognition/`|
|Ethnicity recognition|aligned 112x112 img|probabilities vector 1x4 |`ethnicity_recognition/`|
|Gender recognition|aligned 112x112 img|probabilities vector 1x2|`gender_recognition/`|

# TODO
[ ] - face_angles, предсказывание поворота головы по yaw, pitch, roll
[ ] - Обучить легкие сети для задач age, ethnicity, gender recognition c увеличенной точностью, но с просадкой точности




