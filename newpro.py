import os
import numpy as np

from matplotlib import pyplot
from matplotlib.image import imread

import tensorflow
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
model = tf.keras.models.Sequential([

    # Primeira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Segunda camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Terceira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Quarta camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Achatar (flatten) os resultados para alimentar em uma camada densa
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    # Camada de classificação
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Diretório de imagens de teste
testing_image_directory = 'testing_dataset/infected'

# Todos os arquivos de imagem no diretório
img_files = os.listdir(testing_image_directory)

i= 0

# Loop através de 9 arquivos de imagem
for file in img_files[51:60]:

  # caminho completo da imagem
  img_files_path = os.path.join(testing_image_directory, file)

  # carregar imagem
  img_1 = load_img(img_files_path,target_size=(180, 180))

  # converter imagem para um array
  img_2 = img_to_array(img_1)

  # incrementar a dimensão
  img_3 = np.expand_dims(img_2, axis=0)

  # prever a classe de uma imagem não vista
  prediction = model.predict(img_3)
  # print(prediction)

  predict_class = np.argmax(prediction, axis=1)
  # print(predict_class)

  # plotar a imagem usando subimagens
  pyplot.subplot(3, 3, i+1)
  pyplot.imshow(img_2.astype('uint8'))

  # Adicionar o rótulo da imagem com o valor previsto da classe
  pyplot.title(predict_class[0])

  # Não mostrar eixos x e y com a imagem
  pyplot.axis('off')

  i=i+1

pyplot.show()