from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QWidget, QApplication, QLabel, QGridLayout, QSlider, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import time

from VideoPlayer import VideoWidget

if sys.platform == 'win32':
    current_directory = ''.join([x + '/' for x in os.path.realpath(__file__).split('\\')[:-1]])[:-1]
else:
    current_directory = ''.join([x + '/' for x in os.path.realpath(__file__).split('/')[:-1]])[:-1]

class SlideBar(QWidget):

    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.initUI()

    def initUI(self):

        hbox = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.num_frames)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setPageStep(5)
        self.slider.valueChanged.connect(self.updateLabel)

        self.label = QLabel('0', self)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        hbox.addWidget(self.slider)
        hbox.addSpacing(15)
        hbox.addWidget(self.label)

        self.setLayout(hbox)
        self.show()

    def updateLabel(self, value):
        self.label.setText(str(value))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent, filename, time_series):
        super().__init__()
        self._run_flag = True
        self.parent = parent
        self.filename = filename
        self.time_series = time_series
        self.cap = cv2.VideoCapture(self.filename)
        self.FPS= int(self.cap.get(cv2.CAP_PROP_FPS))
        self.NUM_FRAMES = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = np.array([cv2.resize(self.cap.read()[1], (640, 480)) for iteration in range(self.NUM_FRAMES)])
        self.currentFrame = 0
        self.parent.image_label.setPixmap(self.parent.convert_cv_qt(self.frames[0]))



    def run(self):
        # capture from web cam
        for iteration, frame in enumerate(self.frames):
            self.parent.Slider.slider.setValue(iteration)
            frame = self.process_frame(img = frame, current_index = iteration)
            self.change_pixmap_signal.emit(frame)
            time.sleep(1 / self.FPS)
        # shut down capture system
        self.cap.release()
    
    def process_frame(self, img, current_index):
        b,g,r,a = 0,0,0,0
        img = cv2.rectangle(img, (0, 0), (500, 150), (255, 255, 255), -1)
        ## Use simsum.ttc to write Chinese.
        fontpath = f"{current_directory}/fonts/a_AvanteLt_DemiBoldItalic.ttf"
        try:
            message  = [data for (data, duration) in self.time_series.items() if current_index in range(duration[0], duration[1])][0]
        except IndexError:
            message = list(self.time_series.keys())[0]
        font = ImageFont.truetype(fontpath, 100)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((30, 30),  f"{message}", font = font, fill = (b, g, r, a))
        img = np.array(img_pil)

        return img


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class VideoWidget(QWidget):
    def __init__(self, filename, time_series):
        super().__init__()
        self.filename = filename
        self.time_series = time_series
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(self.disply_width, self.display_height)

        # create a text label
        self.startIconPath = f'{current_directory}/icons/play_button.png'
        self.stopIconPath = f'{current_directory}/icons/stop_button.png'
        self.PlayButton = QPushButton()
        self.PlayButton.setIcon(QtGui.QIcon(QtGui.QPixmap(self.startIconPath)))
        self.PlayButton.setCheckable(True)
        self.PlayButton.clicked.connect(self.play)
        

        # create the video capture thread
        self.thread = VideoThread(parent = self, filename = self.filename, time_series = time_series)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        
        self.Slider = SlideBar(num_frames=self.thread.NUM_FRAMES)


        # create a vertical box layout and add the two labels
        vbox = QGridLayout()
        vbox.addWidget(self.image_label, 0, 0, 0, 2)
        vbox.addWidget(self.PlayButton, 1, 0)
        vbox.addWidget(self.Slider, 1, 1)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

    def preprocess_frame(frame, current_index):
        pass


    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    
    def play(self):
        if self.PlayButton.isChecked():
            self.PlayButton.setIcon(QtGui.QIcon(QtGui.QPixmap(self.stopIconPath)))
            self.thread.start()
        else:
            self.PlayButton.setIcon(QtGui.QIcon(QtGui.QPixmap(self.startIconPath)))
            self.thread.stop() 


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    

filename = 'c:/Users/user/Downloads/data_sign_language/merged_video.mp4'
time_series = {"А": (0, 64), "Б": (64, 128)}
print()
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = VideoWidget(filename = filename, time_series= time_series)
    a.show()
    sys.exit(app.exec_())