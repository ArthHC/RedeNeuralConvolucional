import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Carregar o modelo de rede neural

#rede_neural = load_model('C:/Users/ahcib/OneDrive/Aulas/5º Semestre/IA/Aprendizado por reforço/RedeConvolucional/Models/best_model.keras')
rede_neural = load_model('C:/Users/ahcib/OneDrive/Aulas/5º Semestre/IA/Aprendizado por reforço/RedeConvolucional/Models/EmotionModel.h5')


class PhotoCaptureThread(QThread):
    capture_finished = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.capture_finished.emit(frame)

class PhotoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Photo Capture and Emotion Detection App")
        self.setGeometry(100, 100, 800, 600)

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_and_predict)

        self.emotions = []  # Array para armazenar os resultados das emoções
        self.emotion_times = []  # Array para armazenar os tempos de captura das emoções
        self.start_time = None  # Variável para armazenar o tempo de início

        self.initUI()

    def initUI(self):
        self.start_button = QPushButton("Start Capture", self)
        self.start_button.clicked.connect(self.start_timer)

        self.stop_button = QPushButton("Stop Capture", self)
        self.stop_button.clicked.connect(self.stop_timer)

        self.status_label = QLabel("Status: Stopped", self)
        self.emotion_label = QLabel("Last Emotion: None", self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_timer(self):
        self.timer.start(10000)  # 3 minutos em milissegundos
        self.status_label.setText("Status: Capturing every 3 minutes")
        self.start_time = time.time()  # Registrar o tempo de início

    def stop_timer(self):
        self.timer.stop()
        self.status_label.setText("Status: Stopped")
        self.calculate_and_plot_emotion_percentages()

    def capture_and_predict(self):
        self.photo_thread = PhotoCaptureThread()
        self.photo_thread.capture_finished.connect(self.process_captured_image)
        self.photo_thread.start()

    def process_captured_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        input_image = normalized.reshape(1, 48, 48, 1)

        previsao = rede_neural.predict(input_image)
        emocao = np.argmax(previsao, axis=1)[0]

        emocao_mapeamento = {0: 'Raiva', 1: 'Nojo', 2: 'Medo', 3: 'Felicidade', 4: 'Tristeza', 5: 'Surpresa'}
        emocao_prevista = emocao_mapeamento[emocao]
        print(f"Emoção prevista: {emocao_prevista}")

        self.emotions.append(emocao_prevista)
        self.emotion_times.append(time.time() - self.start_time)  # Registrar o tempo de captura da emoção
        self.emotion_label.setText(f"Last Emotion: {emocao_prevista}")
        self.photo_thread.quit()
        self.photo_thread.wait()

    def calculate_and_plot_emotion_percentages(self):
        total_time = self.emotion_times[-1] if self.emotion_times else 0
        if total_time == 0:
            print("Nenhum tempo de emoção registrado.")
            return

        emotion_counts = {emotion: 0 for emotion in ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa']}
        for emotion in self.emotions:
            emotion_counts[emotion] += 1

        emotion_percentages = {emotion: (count / len(self.emotions)) * 100 for emotion, count in emotion_counts.items()}

        emotions = list(emotion_percentages.keys())
        percentages = list(emotion_percentages.values())

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(emotions, percentages, color=['red', 'green', 'blue', 'yellow', 'cyan', 'purple'])
        ax.set_xlabel('Emoções')
        ax.set_ylabel('Porcentagem de Tempo (%)')
        ax.set_title('Porcentagem de Tempo para Cada Emoção')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoCaptureApp()
    window.show()
    sys.exit(app.exec())
