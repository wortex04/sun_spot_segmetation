# Нейросеть для нахождения солнечных пятен по фотографии солнца
(интерфейс и веса в дальнейшем будут обновлены)

Скачать веса для модели можно здесь:
[U-Net for segmentation weights](https://drive.google.com/file/d/1VoDqbgot3o-DGIV1mvL1cDQZAxM-fvGm/view?usp=drive_link)

Или в notebook с помощью следующей комнады:
```python
!gdown 1VoDqbgot3o-DGIV1mvL1cDQZAxM-fvGm
!unzip unet_weights.zip
```
Скаченные веса следует поместить в папку main_model

## Примеры предсказаний с помощью модкели:


![image](https://github.com/wortex04/sun_spot_segmetation/assets/152957458/cf97f7d8-56fc-4317-9cec-d0eb84d64923)

## Так же можно посмотреть как выглядит маска пятен:


![image](https://github.com/wortex04/sun_spot_segmetation/assets/152957458/8185b734-1af9-431a-816b-cbd8c793eba5)


## Или настроить порог, при котором пиксель счиатется пятном, по умолчанию 0.5:


![image](https://github.com/wortex04/sun_spot_segmetation/assets/152957458/d734c904-206d-4be5-9ec0-0ad8d3eeedc6)

