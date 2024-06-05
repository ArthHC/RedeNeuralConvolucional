import pandas as pd
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import csv

"""## Carregamento das imagens"""

df = pd.read_csv('C:/TesteTrainCSV/train.csv', header=None, quoting=csv.QUOTE_ALL)

def process_cell(cell):
    # Remover aspas e dividir pela primeira vírgula
    parts = cell.replace('"', '').split(',')
    emotion = int(parts[0])  # Converter a emoção para inteiro
    pixel_values = list(map(int, parts[1].split()))  # Converter os valores de pixels para inteiros
    image = np.array(pixel_values).reshape(48, 48)  # Converter a lista de pixels para uma matriz 48x48
    return emotion, image

# Processar todas as células da planilha
labels = []
images = []

for i in range(len(df)):
    for j in range(len(df.columns)):
        # Ignorar células vazias
        if pd.isna(df.iat[i, j]):
            continue
        emotion, image = process_cell(df.iat[i, j])
        labels.append(emotion)
        images.append(image)

X = np.array(images).astype('float32') / 255.0  # Normalizar os valores dos pixels
X = X.reshape(-1, 48, 48, 1)
y = np.array(labels)

# Converter labels para one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=7)

"""## Construção das bases de treinamento e validação"""

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Geradores de dados
gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

# Não utilizamos o flow_from_directory aqui, pois estamos carregando os dados do CSV diretamente
base_treinamento = gerador_treinamento.flow(X_train, y_train, batch_size=32)
base_teste = gerador_teste.flow(X_val, y_val, batch_size=32)

"""## Construção e treinamento da rede neural"""

rede_neural = Sequential()
rede_neural.add(Conv2D(32, (3,3), input_shape = (48,48,1), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Conv2D(64, (3,3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Flatten())

rede_neural.add(Dense(units = 128, activation='relu'))
rede_neural.add(Dense(units = 7, activation='softmax'))

rede_neural.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics = ['accuracy'])


# Treinar o modelo
history = rede_neural.fit(
    base_treinamento,
    epochs=250,
    validation_data=base_teste
)

rede_neural.save('C:/Users/ahcib/OneDrive/Aulas/5º Semestre/IA/Rede neural convolucional/RedeConvolucional/models/that_model.keras')


