import math
from typing import Type
import onnxruntime
import albumentations as albu
import cv2
import PyQt5
from PyQt5 import QtCore, QtGui
import numpy as np
import os
import time
import sys

if sys.platform == 'win32':
    current_directory = ''.join([x + '/' for x in os.path.realpath(__file__).split('\\')[:-1]])[:-1]
else:
    current_directory = ''.join([x + '/' for x in os.path.realpath(__file__).split('/')[:-1]])[:-1]

classes = ['A', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', "Й", 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
           'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', "мама", 'читать', 'книга', 'бабушка', 'вязать', 'голубой',
           'ресторан', 'стоить', 'дорого', 'плавать', 'река', 'зима', 'такси', 'ребенок', 'прогулка', 'собака', 'лето', 'город']

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum


class CFG:  
    '''
    Concfiguration class
    '''
    NUM_FRAMES = 64
    NUM_FRAMES_STREAM = 64
    SIZE = 112
    THRESHOLD = 0.65
    classification_session = onnxruntime.InferenceSession(f'{current_directory}/weights/r2+1d_112_64_main.onnx')
    u2net_5_session = onnxruntime.InferenceSession(f'{current_directory}/weights/u2net_5_360_640.onnx')
    u2net_2_session = onnxruntime.InferenceSession(f'{current_directory}/weights/u2net_2_180_320.onnx')

    first_transform = albu.Compose([albu.Resize(360, 640)])
    first_transform_stream = albu.Compose([albu.Resize(180, 320)])
    second_transform = albu.Compose([
        albu.Resize(SIZE, SIZE),
        albu.CLAHE(p=1),
        albu.Normalize(mean=[0.43216, 0.394666, 0.37645],
                       std=[0.22803, 0.22145, 0.216989])])


class MLManager_MonoValued(PyQt5.QtCore.QThread):
    makePrediction = PyQt5.QtCore.pyqtSignal(str)
    PATH = None

    def run(self): 
        '''
        main function to inference videos which consists 
        1-single sign per video
        '''
        # Чтение кадров
        frames = []
        cap = cv2.VideoCapture(self.PATH)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else: break
        cap.release()

        # Inference U-net
        outputs = []
        for i in range(0, len(frames), len(frames) // 5):
            if len(outputs) < 5:
                image = frames[i]
                image = CFG.first_transform(image=image)['image']
                image = image.transpose(2, 0, 1)
                outputs.append([image])
        outputs = np.concatenate(outputs)
        outputs = self.inf_u2net_5(outputs)[0] * 255
        outputs = outputs.astype(np.uint8)
        binary = np.where(outputs > 10, 255, 0)
        binary[:, :, :, :5] = 0
        binary[:, :, :, -5:] = 0

        # Поиск крайних точек
        min_w, min_h, max_w, max_h = math.inf, math.inf, 0, 0
        for i in range(binary.shape[2]):
            if np.any(binary[:, :, i, :]) and min_h > i: min_h = i
        for i in range(binary.shape[3]):
            if np.any(binary[:, :, :, i]) and min_w > i: min_w = i
        for i in range(binary.shape[2] - 1, -1, -1):
            if np.any(binary[:, :, i, :]) and max_h < i: max_h = i
        for i in range(binary.shape[3] - 1, -1, -1):
            if np.any(binary[:, :, :, i]) and max_w < i: max_w = i

        # Обработка кадров и inference основной сети
        frame_list = []
        part = len(frames) / CFG.NUM_FRAMES
        for i in range(CFG.NUM_FRAMES):
            self.parent.progress_bar.bar.setValue(i + 1)
            frame = frames[int(part * i)]
            frame = CFG.first_transform(image=frame)['image'][min_h:max_h, min_w:max_w]
            frame_list.append(CFG.second_transform(image=frame)['image'])
        frame_list = np.array([frame_list]).transpose((0, 4, 1, 2, 3))
        pred = softmax(self.inf_clas(frame_list))
        prediction = classes[pred.argmax()]
        prediction = prediction if prediction else ''
        self.makePrediction.emit(prediction)

    def inf_clas(self, image):  # Inference основной сети
        ort_inputs = {CFG.classification_session.get_inputs()[0].name: image}
        ort_outs = CFG.classification_session.run(None, ort_inputs)
        return ort_outs[0][0]

    def inf_u2net_5(self, image):  # Inference U-net
        ma = np.max(image)
        mi = np.min(image)
        image = (image - mi) / (ma - mi)
        ort_inputs = {CFG.u2net_5_session.get_inputs()[0].name: image}
        ort_inputs['input.1'] = ort_inputs['input.1'].astype(np.float32)
        ort_outs = CFG.u2net_5_session.run(None, ort_inputs)
        return ort_outs
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.terminate()
    
    def switch_to_MonoValue(self):
        '''activate stream mode'''
        self.makePrediction.connect(self.parent.MonoValuePrediction)
        self.start()










class MLManager_MultiValued(PyQt5.QtCore.QThread):  
    makePrediction = PyQt5.QtCore.pyqtSignal(list)
    PATH = None
    # Главная функция для multiple

    def run(self):
        # Обработка видео
        frames = []
        cap = cv2.VideoCapture(self.PATH)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()

        # Inference U-net
        outputs = []
        for i in range(0, len(frames), len(frames) // 5):
            if len(outputs) < 5:
                image = frames[i]
                image = CFG.first_transform(image=image)['image']
                image = image.transpose(2, 0, 1)
                outputs.append([image])
        outputs = np.concatenate(outputs)
        outputs = self.inf_u2net_5(outputs)[0] * 255
        outputs = outputs.astype(np.uint8)
        binary = np.where(outputs > 10, 255, 0)
        binary[:, :, :, :5] = 0
        binary[:, :, :, -5:] = 0

        # Поиск крайних точек
        min_w, min_h, max_w, max_h = math.inf, math.inf, 0, 0
        for i in range(binary.shape[2]):
            if np.any(binary[:, :, i, :]) and min_h > i: min_h = i
        for i in range(binary.shape[3]):
            if np.any(binary[:, :, :, i]) and min_w > i: min_w = i
        for i in range(binary.shape[2] - 1, -1, -1):
            if np.any(binary[:, :, i, :]) and max_h < i: max_h = i
        for i in range(binary.shape[3] - 1, -1, -1):
            if np.any(binary[:, :, :, i]) and max_w < i: max_w = i

    


        if len(frames) < CFG.NUM_FRAMES:
            tmp_frames = frames
            part = len(frames) / CFG.NUM_FRAMES
            frames = []
            for i in range(CFG.NUM_FRAMES):
                frames.append(tmp_frames[int(part*i)])

        indexes = range(0, len(frames) - CFG.NUM_FRAMES + 1, CFG.NUM_FRAMES // 2)

        # Inference основной сети сквозным окном
        prediction = []
        duration = []

        self.parent.progress_bar.bar.setRange(0, max(max(list(indexes)), 1))
        for i in indexes:
            self.parent.progress_bar.bar.setValue(i + 1)
            pred_frames = frames[i:i+CFG.NUM_FRAMES]
            pred_frames = [CFG.first_transform(image=fr)['image'][min_h:max_h, min_w:max_w] for fr in pred_frames]
            pred_frames = [CFG.second_transform(image=fr)['image'] for fr in pred_frames]
            pred_frames = np.array([pred_frames]).transpose((0, 4, 1, 2, 3))
            pred = softmax(self.inf_clas(pred_frames))
            if pred.max() > 0.5 and (not prediction or prediction[-1] != pred.argmax()):
                prediction.append(pred.argmax())
                duration.append((i, i+CFG.NUM_FRAMES))
        
        prediction = [f'{classes[x]}' for x in prediction]
        time_series = dict(zip(prediction, duration))
        prediction = [f'{pred}, ' for pred in prediction]

        prediction = ''.join(prediction)[:-2] if prediction else ''
        print(time_series)

        self.makePrediction.emit([prediction, time_series])

    # Inference for the main network
    def inf_clas(self, image):  
        ort_inputs = {CFG.classification_session.get_inputs()[0].name: image}
        ort_outs = CFG.classification_session.run(None, ort_inputs)
        return ort_outs[0][0]

    # Inference U-net
    def inf_u2net_5(self, image): 
        ma = np.max(image)
        mi = np.min(image)
        image = (image - mi) / (ma - mi)
        ort_inputs = {CFG.u2net_5_session.get_inputs()[0].name: image}
        ort_inputs['input.1'] = ort_inputs['input.1'].astype(np.float32)
        ort_outs = CFG.u2net_5_session.run(None, ort_inputs)
        return ort_outs
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.terminate()
    
    def switch_to_MonoValue(self):
        '''activate stream mode'''
        self.makePrediction.connect(self.parent.MonoValuePrediction)
        self.start()








class MLManager_Stream(PyQt5.QtCore.QThread):
    makePrediction = PyQt5.QtCore.pyqtSignal()
    frames = list()


    def run(self):  # Основная функция стрима
        # preprocessing frames 
        part = len(self.frames) / CFG.NUM_FRAMES_STREAM
        need_frames = [self.frames[int(part * i)] for i in range(CFG.NUM_FRAMES_STREAM)]
        # Inference U-net

        outputs = []
        for i in range(0, len(need_frames), len(need_frames) // 2):
            if len(outputs) < 2:
                image = need_frames[i]
                image = CFG.first_transform_stream(image=image)['image']
                image = image.transpose(2, 0, 1)
                outputs.append([image])
        outputs = np.concatenate(outputs)
        outputs = self.inf_u2net_5(outputs)[0] * 255
        outputs = outputs.astype(np.uint8)
        binary = np.where(outputs > 10, 255, 0)
        binary[:, :, :, :5] = 0
        binary[:, :, :, -5:] = 0

        cv2.imwrite('bin1.jpg', binary[0, 0, :, :])
        cv2.imwrite('bin2.jpg', binary[1, 0, :, :])

        # extreme points search
        min_w, min_h, max_w, max_h = math.inf, math.inf, 0, 0
        for i in range(binary.shape[2]):
            if np.any(binary[:, :, i, :]) and min_h > i: min_h = i
        for i in range(binary.shape[3]):
            if np.any(binary[:, :, :, i]) and min_w > i: min_w = i
        for i in range(binary.shape[2] - 1, -1, -1):
            if np.any(binary[:, :, i, :]) and max_h < i: max_h = i
        for i in range(binary.shape[3] - 1, -1, -1):
            if np.any(binary[:, :, :, i]) and max_w < i: max_w = i

        if max_w == 0: return False

        # Inference of the main network
        new_frames = []
        for i, frame in enumerate(need_frames):
            frame = CFG.first_transform_stream(image=frame)['image'][min_h:max_h, min_w:max_w]
            new_frames.append(CFG.second_transform(image=frame)['image'])
        new_frames = np.array([new_frames]).transpose((0, 4, 1, 2, 3))
        pred = softmax(self.inf_stream(new_frames))
        prediction = classes[pred.argmax()] if pred.max() > CFG.THRESHOLD else ''
        print(f'{pred.max()} : {classes[pred.argmax()]}\n')
        if not prediction: prediction = ''
        self.parent.prediction = prediction
        if prediction:
            self.parent.history += f'{prediction}, '
        # self.makePrediction.emit(prediction)
        self.makePrediction.emit()
    def inf_u2net_5(self, image):
        '''
        Inference U-net
        '''
        ma = np.max(image)
        mi = np.min(image)
        image = (image - mi) / (ma - mi)
        ort_inputs = {CFG.u2net_5_session.get_inputs()[0].name: image}
        ort_inputs['input.1'] = ort_inputs['input.1'].astype(np.float32)
        ort_outs = CFG.u2net_2_session.run(None, ort_inputs)
        return ort_outs

    def inf_stream(self, image):  # Inference основной сети
        ort_inputs = {CFG.classification_session.get_inputs()[0].name: image}
        ort_outs = CFG.classification_session.run(None, ort_inputs)
        return ort_outs

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        # self._run_flag = False
        # self.terminate()
        pass
    
    # def switchToStream(self):
        # '''activate stream mode'''
        # self.makePrediction.connect(self.parent.get_prediction)
        # self.start()
    
    
