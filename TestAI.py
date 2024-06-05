import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def process_cell(cell):
    parts = cell.replace('"', '').split(',')
    emotion = int(parts[0])  # Converter a emoção para inteiro
    pixel_values = list(map(int, parts[1].split()))  # Converter os valores de pixels para inteiros
    image = np.array(pixel_values).reshape(48, 48)  # Converter a lista de pixels para uma matriz 48x48
    return emotion, image

# Carregar dados do CSV
df = pd.read_csv('C:/Users/ahcib/OneDrive/Aulas/5º Semestre/IA/Rede neural convolucional/test/TestedoTrain.csv', header=None, quoting=csv.QUOTE_ALL)

# Ignorar o cabeçalho
df = df.iloc[1:]

# Processar todas as células da planilha
labels = []
images = []

for i in range(len(df)):
    for j in range(len(df.columns)):
        if pd.isna(df.iat[i, j]):
            continue
        emotion, image = process_cell(df.iat[i, j])
        labels.append(emotion)
        images.append(image)

X = np.array(images).astype('float32') / 255.0  # Normalizar os valores dos pixels
X = X.reshape(-1, 48, 48, 1)
y = np.array(labels)
y = tf.keras.utils.to_categorical(y, num_classes=7)

# Dividir os dados em treinamento (80%) e teste (20%)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Carregar o melhor modelo salvo
melhor_modelo = tf.keras.models.load_model('C:/Users/ahcib/OneDrive/Aulas/5º Semestre/IA/Rede neural convolucional/RedeConvolucional/models/that_model.keras')

# Fazer previsões com o modelo treinado
previsoes = melhor_modelo.predict(X_test)
previsoes2 = np.argmax(previsoes, axis=1)

# Verificar as classes reais
classes_reais = np.argmax(y_test, axis=1)

# Avaliar a acurácia
acuracia = accuracy_score(previsoes2, classes_reais)
print(f'Acurácia: {acuracia:.2f}')

# Matriz de confusão
cm = confusion_matrix(classes_reais, previsoes2)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')
plt.show()
