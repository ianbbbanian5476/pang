import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import threading
import cv2
import shutil
import os
import time
import configparser
from functools import partial
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
from ultralytics import YOLO
import numpy as np
from math import dist
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# 根據麵包的曝光值畫熱力圖
def draw_saturation_heat_map(image):
    # 將圖片轉成HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 抽出曝光層
    saturation_channel = hsv_image[:, :, 1]
    saturation_channel = saturation_channel / 255

    # 根據曝光值畫熱力圖
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(saturation_channel, cmap='jet', vmin=0, vmax=0.8)
    ax.axis('off')
    fig.canvas.draw()

    # 為了後面要把熱力圖顯示在介面上，先將熱力圖轉成numpy array的格式
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    saturation_heat_map = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    saturation_heat_map = cv2.cvtColor(saturation_heat_map, cv2.COLOR_RGB2BGR)

    # 清除畫布
    plt.cla()
    plt.close('all')

    return saturation_heat_map

# 將圖片抽出曝光值後，根據曝光值做出直方圖
def saturation_histogram(image):
    # 將圖片轉成HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat_image = hsv_image[:, :, 1]  # 抽出曝光值的層

    # 只留下麵包範圍內的pixels，並把pixel數值都轉成0~1之間
    non_transparent_pixels = sat_image[np.where(gray_image != 0)]
    non_transparent_pixels = non_transparent_pixels / 255

    # 計算並返回直方圖，後續訓練模型時用的特徵數量就會是bins的數量
    histogram, bin_edges = np.histogram(non_transparent_pixels, bins=5, range=(0, 1))
    return histogram, bin_edges

# 讓使用者點選哪張圖片算烘焙好的圖片，那個可點選的圖片就是從這個class下去生成的
class ImageLabel(QLabel):
    image_label_clicked_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMaximumWidth(640)
        self.setMaximumHeight(360)
        self.setScaledContents(True)

    def mouseReleaseEvent(self, QMouseEvent):
        self.image_label_clicked_signal.emit()
    
    def image_label_clicked(self, func):
        self.image_label_clicked_signal.connect(func)

# 新增烘焙模式錄影完成後，會開啟這個視窗給使用者填麵包名稱及選擇烘焙完成的圖片
class InputNewModeDataWindow(QWidget):
    move_data_start_signal = pyqtSignal()
    move_data_end_signal = pyqtSignal()
    segment_data_start_signal = pyqtSignal()
    segment_data_end_signal = pyqtSignal()
    train_model_start_signal = pyqtSignal(int)
    train_model_end_signal = pyqtSignal()

    def __init__(self, images_dir):
        super().__init__()
        uic.loadUi("input_new_mode_data.ui", self) # 這個視窗的外觀是根據這個ui檔生成的
        self.current_selected_index = 0
        self.images_dir = images_dir
        self.music_path = os.path.join("musics", "kit_timer_beep.mp3") # 模型訓練完成時會去撥放這個mp3檔
        self.play_music_thread = None

        # 初始化存放圖片的List
        self.imageLabelScrollAreaLayout.setAlignment(Qt.AlignCenter)
        self.image_label_list = []

        # 設定確定按鈕要觸發的功能
        self.confirmButton.clicked.connect(self.confirm_button)

        # 設定不同階段開始及結束時會觸發的功能
        self.move_data_start_signal.connect(self.move_data_start_listener)
        self.move_data_end_signal.connect(self.move_data_end_listener)
        self.segment_data_start_signal.connect(self.segment_data_start_listener)
        self.segment_data_end_signal.connect(self.segment_data_end_listener)
        self.train_model_start_signal.connect(self.train_model_start_listener)
        self.train_model_end_signal.connect(self.train_model_end_listener)

    # 將圖片加進給使用者選擇的圖片List內
    def add_image_label(self, index, qimage):
            image_label = ImageLabel(f'{index}')
            image_label.setPixmap(QPixmap.fromImage(qimage))
            image_label.image_label_clicked(partial(self.image_label_clicked, index))
            self.imageLabelScrollAreaLayout.addWidget(image_label)
            self.image_label_list.append(image_label)

    # 點選圖片後會觸發的動作
    def image_label_clicked(self, index):
        for image_label in self.image_label_list:
            image_label.setStyleSheet('''
                border: 0px;
            ''')
        self.image_label_list[index].setStyleSheet('''
            border: 5px groove green;
        ''')
        self.current_selected_index = index

    # 按確認按鈕後會觸發的動作
    def confirm_button(self):
        bread_name = self.breadNameLineEdit.text()
        if bread_name == '':
            QMessageBox.warning(None, '資料缺漏', '未輸入麵包名稱！')
        else:
            self.confirmButton.setEnabled(False)
            self.camera_thread = threading.Thread(target = self.confirm_button_job)
            self.camera_thread.start()
    
    # 為了不卡住介面，點選確定後會額外用一個thread執行這段程式碼來訓練模型
    def confirm_button_job(self):
        # 先將需要創建的資料夾建好
        mode_dir = 'modes'
        if not os.path.isdir(mode_dir):
            os.mkdir(mode_dir)

        bread_name = self.breadNameLineEdit.text()
        bread_dir = os.path.join(mode_dir, bread_name)
        if os.path.isdir(bread_dir):
            shutil.rmtree(bread_dir)
        os.mkdir(bread_dir)

        not_ready_dir = os.path.join(bread_dir, 'not_ready')
        ready_dir = os.path.join(bread_dir, 'ready')
        os.mkdir(not_ready_dir)
        os.mkdir(ready_dir)
        
        # 根據使用者選擇的圖片，將錄影時錄的圖片移動到ready或not_ready的資料夾
        self.move_data_start_signal.emit()
        for image_name in os.listdir(self.images_dir):
            source_image_path = os.path.join(self.images_dir, image_name)
            if int(image_name[:-4]) < self.current_selected_index:
                destination_image_path = os.path.join(not_ready_dir, image_name)
            else:
                destination_image_path = os.path.join(ready_dir, image_name)
            shutil.move(source_image_path, destination_image_path)
        self.move_data_end_signal.emit()

        # 發出一個代表即將開始分割模型的訊號
        self.segment_data_start_signal.emit()

        # 載入分割麵包用的模型
        segment_model_path = 'segment_models/bread-seg.pt'
        model = YOLO(segment_model_path)
        x = []
        y = []

        last_ready_image_number = 0
        last_ready_histogram = None
        last_not_ready_image_number = 0
        last_not_ready_histogram = None
        is_first_segment = True
        last_bounding_box_center = None

        # 處理not_ready內的麵包
        for image_name in os.listdir(not_ready_dir):
            image_path = os.path.join(not_ready_dir, image_name)
            image = cv2.imread(image_path)

            # 用分割模型預測麵包的範圍
            results = model(image)  # predict on an image
            result = results[0]

            # 處理預測結果
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            probs = result.probs  # Probs object for classification outputs

            objects_len = len(result)
            if objects_len == 0:
                continue

            # 一張圖內有很多顆麵包，我們只會挑其中一顆麵包做後續的曝光值轉直方圖的動作，
            # 而我們希望能盡量挑靠中間且比較大顆的麵包，所以有再來的這些處理，
            # 第一次分割時會先計算面積最大的3個麵包，再從3個麵包取最靠近畫面中間的麵包。
            if is_first_segment:
                # 紀錄每個分割出來的麵包bounding_box和mask還有面積
                target_list = []
                for i in range(objects_len):
                    bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                    mask_contour = np.array(masks.xy[i], dtype=int)
                    contour_area = cv2.contourArea(mask_contour)
                    target_list.append((bounding_box, mask_contour, contour_area))

                # 根據面積做排序
                for i in range(objects_len):
                    for j in range(i + 1, objects_len):
                        if target_list[i][2] < target_list[j][2]:
                            temp = target_list[j]
                            target_list[j] = target_list[i]
                            target_list[i] = temp

                # 取出面積最大的三個麵包
                target_list = target_list[:3]

                # 從這三個麵包挑出最接近中間的麵包
                target_bounding_box = None
                target_contour = None
                target_distance = 999999
                frame_center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
                for target_data in target_list:
                    bounding_box = target_data[0]
                    mask_contour = target_data[1]
                    contour_area = target_data[2]
                    bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                    center_distance = dist(bounding_box_center, frame_center)
                    if center_distance < target_distance:
                        target_bounding_box = bounding_box
                        target_contour = mask_contour
                        target_distance = center_distance
                # 紀錄最靠近中間的麵包位置
                last_bounding_box_center = (int((target_bounding_box[0] + target_bounding_box[2]) / 2), int((target_bounding_box[1] + target_bounding_box[3]) / 2))
                is_first_segment = False
            # 後續都是看這個frame哪一個麵包跟上一個frame挑選的麵包距離最接近
            else:
                # 計算哪一個麵包距離上一個frame挑選的麵包距離最接近
                target_bounding_box = None
                target_contour = None
                target_distance = 999999
                target_bounding_box_center = None
                for i in range(objects_len):
                    bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                    mask_contour = np.array(masks.xy[i], dtype=int)
                    contour_area = cv2.contourArea(mask_contour)
                    bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                    center_distance = dist(bounding_box_center, last_bounding_box_center)
                    if center_distance < target_distance:
                        target_bounding_box = bounding_box
                        target_contour = mask_contour
                        target_distance = center_distance
                        target_bounding_box_center = bounding_box_center
                last_bounding_box_center = target_bounding_box_center

            # 將挑選的麵包區域擷取出來，然後轉曝光值做直方圖
            black_canvas = np.zeros_like(image)
            cv2.drawContours(black_canvas, [target_contour], -1, (255, 255, 255), cv2.FILLED) # this gives a binary mask
            black_canvas[np.where(black_canvas != 0)] = image[np.where(black_canvas != 0)]
            processed_image = black_canvas[target_bounding_box[1]:target_bounding_box[3], target_bounding_box[0]: target_bounding_box[2]]
            histogram, bin_edges = saturation_histogram(processed_image)

            # 這邊會去紀錄not_ready裡最後一張圖片的histogram，為了後續跟ready裡最後一張圖片的histogram做比較，
            # 如果not_ready跟ready的最後一張圖片的histogram很接近，我就會把判定烘焙完成的機率門檻值調低
            image_number = int(image_name[:-4])
            if image_number > last_not_ready_image_number:
                last_not_ready_image_number = image_number
                last_not_ready_histogram = histogram

            # 準備訓練模型用的資料
            x.append(histogram)
            y.append(0)

        # 處理ready內的麵包
        for image_name in os.listdir(ready_dir):
            image_path = os.path.join(ready_dir, image_name)
            image = cv2.imread(image_path)

            # 用分割模型預測麵包的範圍
            results = model(image)  # predict on an image
            result = results[0]

            # 處理預測結果
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            probs = result.probs  # Probs object for classification outputs

            objects_len = len(result)
            if objects_len == 0:
                continue

            # 計算哪一個麵包距離上一個frame挑選的麵包距離最接近
            target_bounding_box = None
            target_contour = None
            target_distance = 999999
            target_bounding_box_center = None
            for i in range(objects_len):
                bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                mask_contour = np.array(masks.xy[i], dtype=int)
                contour_area = cv2.contourArea(mask_contour)
                bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                center_distance = dist(bounding_box_center, last_bounding_box_center)
                if center_distance < target_distance:
                    target_bounding_box = bounding_box
                    target_contour = mask_contour
                    target_distance = center_distance
                    target_bounding_box_center = bounding_box_center
            last_bounding_box_center = target_bounding_box_center

            # 將挑選的麵包區域擷取出來，然後轉曝光值做直方圖
            black_canvas = np.zeros_like(image)
            cv2.drawContours(black_canvas, [target_contour], -1, (255, 255, 255), cv2.FILLED) # this gives a binary mask
            black_canvas[np.where(black_canvas != 0)] = image[np.where(black_canvas != 0)]
            processed_image = black_canvas[target_bounding_box[1]:target_bounding_box[3], target_bounding_box[0]: target_bounding_box[2]]
            histogram, bin_edges = saturation_histogram(processed_image)

            # 這邊會去紀錄ready裡最後一張圖片的histogram
            image_number = int(image_name[:-4])
            if image_number > last_ready_image_number:
                last_ready_image_number = image_number
                last_ready_histogram = histogram

            # 準備訓練模型用的資料
            x.append(histogram)
            y.append(1)

        # 發出代表分割完成的訊號
        self.segment_data_end_signal.emit()

        # 發出代表開始訓練模型的訊號
        self.train_model_start_signal.emit(len(x))

        # 對直方圖做標準化
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # 訓練SVM模型
        x_train, y_train = x_scaled, y

        svc_model = SVC(kernel='linear', class_weight='balanced', probability=True)
        svc_model.fit(x_train, y_train)

        model_dir = os.path.join(bread_dir, 'models')
        os.mkdir(model_dir)

        # 儲存標準化用的模型
        scaler_model_path = os.path.join(model_dir, 'scaler_model.pkl')
        with open(scaler_model_path, 'wb') as f:  # open a text file
            pickle.dump(scaler, f) # serialize the list

        # 儲存SVM模型
        svc_model_path = os.path.join(model_dir, 'svc_model.pkl')
        with open(svc_model_path, 'wb') as f:  # open a text file
            pickle.dump(svc_model, f) # serialize the list

        # 創建寫有機率值門檻的設定檔，機率值門檻平常會設定在0.9，
        # 代表烘焙完成的機率要大於90%才會判烘焙完成，
        # 不過ready和not_ready最後一張圖差距太小的話，
        # 我會把機率值下調到0.75或是0.5
        last_ready_histogram = scaler.transform([last_ready_histogram])[0]
        last_not_ready_histogram = scaler.transform([last_not_ready_histogram])[0]
        histogram_difference_value = 0
        for i in range(len(last_ready_histogram)):
            histogram_difference_value += abs(last_ready_histogram[i] - last_not_ready_histogram[i])
        if histogram_difference_value < 0.5:
            probability_threshold = 0.5
        elif histogram_difference_value >= 0.5 and histogram_difference_value <= 1:
            probability_threshold = 0.75
        elif histogram_difference_value > 1:
            probability_threshold = 0.9
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'probability_threshold': probability_threshold}
        with open(os.path.join(bread_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)

        
        print(f'Histogram Difference: {histogram_difference_value}')

        # 發出代表模型訓練完成的訊號
        self.train_model_end_signal.emit()

    # 圖片移動開始時會執行這段
    def move_data_start_listener(self):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 開始移動錄製圖片...\n'
        self.textBrowser.append(output_text)

    # 圖片移動完成時會執行這段
    def move_data_end_listener(self):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 圖片移動完成！\n'
        self.textBrowser.append(output_text)

    # 分割開始時會執行這段
    def segment_data_start_listener(self):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 開始分割圖片...\n'
        self.textBrowser.append(output_text)

    # 分割結束時會執行這段
    def segment_data_end_listener(self):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 圖片分割完成！\n'
        self.textBrowser.append(output_text)

    # 模型訓練開始時會執行這段
    def train_model_start_listener(self, value):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 總共 {value} 筆訓練資料，開始訓練模型...\n'
        self.textBrowser.append(output_text)

    # 模型訓練結束時會執行這段
    def train_model_end_listener(self):
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 模型訓練完成！\n'
        self.textBrowser.append(output_text)
        self.play_music_thread = threading.Thread(target = self.play_music).start()
        QMessageBox.information(None, '創建程序結束', '新模式創創建成功！')

    # 撥放烘焙完成的音樂
    def play_music(self):
        song = AudioSegment.from_mp3(self.music_path)
        output = song * 3
        play(output)


# 點選新增烘焙模式後會開啟的視窗
class CreateNewModeWindow(QWidget):

    open_camera_success_signal = pyqtSignal()
    update_image_signal = pyqtSignal()
    add_image_label_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        uic.loadUi("create_new_mode.ui", self)  # 這個視窗的外觀是根據這個ui檔生成的
        self.current_frame = None
        self.current_QImage = None
        self.is_over = False
        self.images_dir = "images"
        self.openCameraButton.setEnabled(True)
        self.startRecordButton.setEnabled(False)
        self.stopRecordButton.setEnabled(False)
        self.openCameraButton.clicked.connect(self.open_camera_button)
        self.startRecordButton.clicked.connect(self.start_record_button)
        self.stopRecordButton.clicked.connect(self.stop_record_button)
        self.open_camera_success_signal.connect(self.open_camera_success_listener)
        self.update_image_signal.connect(self.update_image_listener)
        self.add_image_label_signal.connect(self.add_image_label_listener)

    # 點擊開啟相機後會執行這段
    def open_camera_button(self):
        self.openCameraButton.setEnabled(False)
        self.startRecordButton.setEnabled(True)
        self.camera_thread = threading.Thread(target = self.open_camera_job)
        self.camera_thread.start()
    
    # 為了避免卡住介面，點擊開啟相機後會額外開thread執行取像的動作
    def open_camera_job(self):
        self.is_over = False
        # 以下可以選擇連接不同的影像來源，請根據需要自行註解或反註解後續幾行
    
        # 根據介面輸入的IP選擇連接的手機網址
        # ip_address = 'http://' + self.CamIP.text() + ':8080/videofeed'
        # self.cap = cv2.VideoCapture(ip_address)

        # 直接指定要連接的手機網址
        # self.cap = cv2.VideoCapture('http://10.10.10.116:8080/videofeed')

        # 直接指定要連接的影片
        self.cap = cv2.VideoCapture('test_videos/20240502_175619.mp4')

        # 直接指定要連接的鏡頭
        # self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        # 連接鏡頭成功會觸發這個訊號
        self.open_camera_success_signal.emit()

        # 再來會持續讀取影像來源，直到使用者按停止錄製或關閉視窗
        while not self.is_over:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            self.current_frame = frame
            self.current_QImage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.update_image_signal.emit() # 觸發介面更新圖片
            cv2.waitKey(1000)
        self.cap.release()

    # 按開始錄製後會執行這段
    def start_record_button(self):
        self.startRecordButton.setEnabled(False)
        self.stopRecordButton.setEnabled(True)
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove green;
        ''')
        self.input_new_mode_data_window = InputNewModeDataWindow(self.images_dir)
        self.record_thread = threading.Thread(target = self.start_record_job)
        self.record_thread.start()
    
    # 為了避免卡住介面，點擊開始錄製後會額外開thread執行錄影的動作
    def start_record_job(self):
        # 這邊還會另外錄製一支影片方便後續作一些研究
        output_dir = 'outputs'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_shape = (1920, 1080)
        fps = 1
        datetime_string = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(output_dir, f'{datetime_string}.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, video_shape)

        count = 0
        delay_time = 1 # 設定每經過多久取一張圖
        last_save_time = time.time()

        # 刪除舊的圖片暫存資料夾並重新建立
        if os.path.isdir(self.images_dir):
            shutil.rmtree(self.images_dir)
        os.mkdir(self.images_dir)

        # 每經過上面設定的delay_time，就會存一張圖到指定的暫存位置
        while not self.is_over:
            if time.time() - last_save_time > delay_time:
                self.add_image_label_signal.emit(count)
                image_path = os.path.join(self.images_dir, f"{count}.jpg")
                cv2.imwrite(image_path, self.current_frame)
                out.write(self.current_frame)
                count += 1
                last_save_time = time.time()
            time.sleep(0.01)
        out.release()
            
    # 按停止錄製後會執行這段
    def stop_record_button(self):
        self.openCameraButton.setEnabled(True)
        self.stopRecordButton.setEnabled(False)
        self.is_over = True
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove red;
        ''')
        self.input_new_mode_data_window.image_label_list[0].setStyleSheet('''
            border: 5px groove green;
        ''')
        self.input_new_mode_data_window.show()

    # 開啟相機成功後會執行這段
    def open_camera_success_listener(self):
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove yellow;
        ''')

    # 為了要持續更新介面上的圖片，開啟相機後會不斷呼叫這段程式碼
    def update_image_listener(self):
        self.cameraImageLabel.setPixmap(QPixmap.fromImage(self.current_QImage))

    # 把取到的圖片加到後續讓使用者選擇烘焙完成圖片的介面上
    def add_image_label_listener(self, index):
        self.input_new_mode_data_window.add_image_label(index, self.current_QImage)

    # 關閉視窗時會執行這段，讓前面有使用到無限迴圈的地方中斷
    def closeEvent(self, event):
        self.is_over = True

# 點選智慧烘焙後會開啟的介面
class SmartBakeWindow(QWidget):
    
    open_camera_success_signal = pyqtSignal()
    update_image_signal = pyqtSignal()
    update_heat_map_signal = pyqtSignal()
    bake_complete_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        uic.loadUi("smart_bake.ui", self)  # 這個視窗的外觀是根據這個ui檔生成的
        self.current_frame = None
        self.current_QImage = None
        self.current_mode = None
        self.current_heat_map_QImage = None
        self.probability_threshold = 0
        self.is_over = False
        self.mode_dir = 'modes'
        self.music_path = os.path.join("musics", "kit_timer_beep.mp3")  # 判定烘焙完成時會撥放這個mp3檔
        self.play_music_thread = None

        self.openCameraButton.setEnabled(True)
        self.startDetectButton.setEnabled(False)
        self.stopDetectButton.setEnabled(False)

        self.openCameraButton.clicked.connect(self.open_camera_button)
        self.startDetectButton.clicked.connect(self.start_detect_button)
        self.stopDetectButton.clicked.connect(self.stop_detect_button)

        # 將目前使用者訓練好的所有模式加進ComboBox內
        self.modeComboBox.currentTextChanged.connect(self.mode_change)
        for mode_name in os.listdir(self.mode_dir):
            self.modeComboBox.addItem(mode_name)

        self.open_camera_success_signal.connect(self.open_camera_success_listener)
        self.update_image_signal.connect(self.update_image_listener)
        self.update_heat_map_signal.connect(self.update_heat_map_listener)
        self.bake_complete_signal.connect(self.bake_complete_listener)

    # 點選開啟相機後會執行這段
    def open_camera_button(self):
        self.openCameraButton.setEnabled(False)
        self.startDetectButton.setEnabled(True)
        self.camera_thread = threading.Thread(target = self.open_camera_job)
        self.camera_thread.start()
    
    # 為了避免卡住介面，點擊開啟相機後會額外開thread執行取像的動作
    def open_camera_job(self):
        self.is_over = False
        # 以下可以選擇連接不同的影像來源，請根據需要自行註解或反註解後續幾行
    
        # 根據介面輸入的IP選擇連接的手機網址
        # ip_address = 'http://' + self.CamIP.text() + ':8080/videofeed'
        # self.cap = cv2.VideoCapture(ip_address)

        # 直接指定要連接的手機網址
        # self.cap = cv2.VideoCapture('http://10.10.10.116:8080/videofeed')

        # 直接指定要連接的影片
        self.cap = cv2.VideoCapture('test_videos/20240502_181034.mp4')

        # 直接指定要連接的鏡頭
        # self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        # 連接鏡頭成功會觸發這個訊號
        self.open_camera_success_signal.emit()

        # 再來會持續讀取影像來源，直到使用者按停止錄製或關閉視窗
        while not self.is_over:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            self.current_frame = frame
            self.current_QImage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.update_image_signal.emit() # 觸發介面更新圖片
            cv2.waitKey(1000)
        self.cap.release()

    # 按開始檢測後會執行這段
    def start_detect_button(self):
        self.startDetectButton.setEnabled(False)
        self.stopDetectButton.setEnabled(True)
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove green;
        ''')
        self.detect_thread = threading.Thread(target = self.start_detect_job)
        self.detect_thread.start()
        
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 開始檢測...\n'
        self.textBrowser.append(output_text)
    
    # 為了避免卡住介面，點擊開始檢測後會額外開thread執行檢測的動作
    def start_detect_job(self):
        model_dir = os.path.join(self.mode_dir, self.current_mode, 'models')

        # 載入分割麵包用的模型
        segment_model_path = 'segment_models/bread-seg.pt'
        segment_model = YOLO(segment_model_path)

        # 載入標準化用的模型
        scaler_model_path = os.path.join(model_dir, 'scaler_model.pkl')
        with open(scaler_model_path, 'rb') as f:  # open a text file
            scaler = pickle.load(f) # serialize the list

        # 載入SVM的模型
        svc_model_path = os.path.join(model_dir, 'svc_model.pkl')
        with open(svc_model_path, 'rb') as f:  # open a text file
            svc_model = pickle.load(f) # serialize the list

        # 這邊還會另外錄製一支影片方便後續作一些研究
        output_dir = 'outputs'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_shape = (1920, 1080)
        fps = 1
        datetime_string = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(output_dir, f'{datetime_string}.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, video_shape)

        delay_time = 1  # 設定每經過多久取一張圖
        last_detect_time = time.time()
        is_first_segment = True
        last_bounding_box_center = None

        # 每經過上面設定的delay_time，就會檢測一次
        while not self.is_over:
            if time.time() - last_detect_time > delay_time:
                image = self.current_frame.copy()

                out.write(image)

                # 用分割模型預測麵包的範圍
                results = segment_model(image)  # predict on an image
                result = results[0]

                # 處理預測結果
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                probs = result.probs  # Probs object for classification outputs
                
                objects_len = len(result)
                if objects_len == 0:
                    continue

                # 一張圖內有很多顆麵包，我們只會挑其中一顆麵包做後續的曝光值轉直方圖的動作，
                # 而我們希望能盡量挑靠中間且比較大顆的麵包，所以有再來的這些處理，
                # 第一次分割時會先計算面積最大的3個麵包，再從3個麵包取最靠近畫面中間的麵包。
                if is_first_segment:
                    # 紀錄每個分割出來的麵包bounding_box和mask還有面積
                    target_list = []
                    for i in range(objects_len):
                        bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                        mask_contour = np.array(masks.xy[i], dtype=int)
                        contour_area = cv2.contourArea(mask_contour)
                        target_list.append((bounding_box, mask_contour, contour_area))

                    # 根據面積做排序
                    for i in range(objects_len):
                        for j in range(i + 1, objects_len):
                            if target_list[i][2] < target_list[j][2]:
                                temp = target_list[j]
                                target_list[j] = target_list[i]
                                target_list[i] = temp

                    # 取出面積最大的三個麵包
                    target_list = target_list[:3]

                    # 從這三個麵包挑出最接近中間的麵包
                    target_bounding_box = None
                    target_contour = None
                    target_distance = 999999
                    frame_center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
                    for target_data in target_list:
                        bounding_box = target_data[0]
                        mask_contour = target_data[1]
                        contour_area = target_data[2]
                        bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                        center_distance = dist(bounding_box_center, frame_center)
                        if center_distance < target_distance:
                            target_bounding_box = bounding_box
                            target_contour = mask_contour
                            target_distance = center_distance

                    # 紀錄最靠近中間的麵包位置
                    last_bounding_box_center = (int((target_bounding_box[0] + target_bounding_box[2]) / 2), int((target_bounding_box[1] + target_bounding_box[3]) / 2))
                    is_first_segment = False
                # 後續都是看這個frame哪一個麵包跟上一個frame挑選的麵包距離最接近
                else:
                    # 計算哪一個麵包距離上一個frame挑選的麵包距離最接近
                    target_bounding_box = None
                    target_contour = None
                    target_distance = 999999
                    target_bounding_box_center = None
                    for i in range(objects_len):
                        bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                        mask_contour = np.array(masks.xy[i], dtype=int)
                        contour_area = cv2.contourArea(mask_contour)
                        bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                        center_distance = dist(bounding_box_center, last_bounding_box_center)
                        if center_distance < target_distance:
                            target_bounding_box = bounding_box
                            target_contour = mask_contour
                            target_distance = center_distance
                            target_bounding_box_center = bounding_box_center
                    last_bounding_box_center = target_bounding_box_center

                # 將挑選的麵包區域擷取出來，然後轉曝光值做直方圖
                black_canvas = np.zeros_like(image)
                cv2.drawContours(black_canvas, [target_contour], -1, (255, 255, 255), cv2.FILLED) # this gives a binary mask
                black_canvas[np.where(black_canvas != 0)] = image[np.where(black_canvas != 0)]
                processed_image = black_canvas[target_bounding_box[1]:target_bounding_box[3], target_bounding_box[0]: target_bounding_box[2]]
                saturation_heat_map = draw_saturation_heat_map(processed_image)
                histogram, bin_edges = saturation_histogram(processed_image)
                
                # 這邊會去生成麵包曝光值的熱力圖並更新到介面上
                width_ratio = 640 / saturation_heat_map.shape[1]
                height_ratio = 360 / saturation_heat_map.shape[0]
                if width_ratio >= height_ratio:
                    target_ratio = width_ratio - 0.1
                else: 
                    target_ratio = height_ratio - 0.1
                target_width = int(saturation_heat_map.shape[1] * target_ratio)
                target_height = int(saturation_heat_map.shape[0] * target_ratio)
                saturation_heat_map = cv2.resize(saturation_heat_map, (target_width, target_height), interpolation=cv2.INTER_AREA)
                self.current_heat_map_QImage = QImage(saturation_heat_map.data, saturation_heat_map.shape[1], saturation_heat_map.shape[0], QImage.Format_RGB888).rgbSwapped()
                self.update_heat_map_signal.emit()

                # 對曝光值的直方圖座標準化
                x_scaled = scaler.transform([histogram])

                # 將標準化後的曝光值直方圖給SVM模型判斷是否烘焙完成
                classified_results = svc_model.predict_proba(x_scaled)
                bread_ready_proba = classified_results[0][1]
                print(f'Bread_Ready_Proba: {bread_ready_proba}')

                # 如果烘焙完成的機率大於設定的機率門檻值，就會觸發烘焙完成的訊號
                if bread_ready_proba > self.probability_threshold:
                    self.bake_complete_signal.emit()

                last_detect_time = time.time()
            time.sleep(0.01)
        out.release()

    # 按停止檢測就會執行這段
    def stop_detect_button(self):
        self.openCameraButton.setEnabled(True)
        self.stopDetectButton.setEnabled(False)
        self.is_over = True
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove red;
        ''')

    # 當麵包模式切換時會自動執行這段
    def mode_change(self, mode):
        self.current_mode = mode
        ready_dir = os.path.join(self.mode_dir, mode, 'ready')
        for image_name in os.listdir(ready_dir):
            image_path = os.path.join(ready_dir, image_name)
            image = cv2.imread(image_path)
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.modeImageLabel.setPixmap(QPixmap.fromImage(qimage))
            break
        config = configparser.ConfigParser()
        config.read(os.path.join(self.mode_dir, mode, 'config.ini'))
        self.probability_threshold = float(config['DEFAULT']['probability_threshold'])

    # 當相機連接成功執行這段
    def open_camera_success_listener(self):
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove yellow;
        ''')

    # 為了要持續更新介面上的圖片，開啟相機後會不斷呼叫這段程式碼
    def update_image_listener(self):
        self.cameraImageLabel.setPixmap(QPixmap.fromImage(self.current_QImage))

    # 為了要持續更新介面上的麵包熱力圖，開始檢測後會不斷呼叫這段程式碼
    def update_heat_map_listener(self):
        self.resultImageLabel.setPixmap(QPixmap.fromImage(self.current_heat_map_QImage))

    # 模型判斷烘焙完成後會執行這段
    def bake_complete_listener(self):
        self.openCameraButton.setEnabled(True)
        self.stopDetectButton.setEnabled(False)
        self.is_over = True
        self.cameraImageLabel.setStyleSheet('''
            border: 5px groove red;
        ''')
        datetime_string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:"
        output_text = f'{datetime_string} 烘焙完成！\n'
        self.textBrowser.append(output_text)
        self.play_music_thread = threading.Thread(target = self.play_music).start()
        QMessageBox.information(None, '檢測結束', '烘焙完成！')

    # 烘焙完成時為了撥放音樂會呼叫這段
    def play_music(self):
        song = AudioSegment.from_mp3(self.music_path)
        output = song * 3
        play(output)

    # 關閉視窗時會執行這段，讓前面有使用到無限迴圈的地方中斷
    def closeEvent(self, event):
        self.is_over = True

# 執行bread_bake_app.py後會開啟的視窗
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("bread_bake_app.ui", self)  # 這個視窗的外觀是根據這個ui檔生成的
        self.createNewModeButton.clicked.connect(self.create_new_mode_button)
        self.smartBakeButton.clicked.connect(self.smart_bake_button)

    # 點選新增烘焙模式後會開啟對應的介面
    def create_new_mode_button(self):
        self.create_new_mode_window = CreateNewModeWindow()
        self.create_new_mode_window.show()

    # 點選智慧烘焙後會開啟對應的介面
    def smart_bake_button(self):
        self.smart_bake_window = SmartBakeWindow()
        self.smart_bake_window.show()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()