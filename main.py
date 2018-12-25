import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
#from PIL import ImageFile
#from CNNLOAD import predict_single_image, model_ft





class GUI(QDialog):

    def __init__(self):
        super(GUI, self).__init__()
        loadUi('GUI_layout.ui',self)
        self.image=None
        self.capture = cv2.VideoCapture(0)


        self.captureButton.clicked.connect(self.capture_image)
        self.predictButton.clicked.connect(self.predict_number)
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.loadButton.clicked.connect(self.loadClicked)
        self.PredictedResult.setText('Hello')



        #self.pred = ''


    def capture_image(self):
        frame = self.capture.read()[1]
        cv2.imwrite(filename='temp.jpg', img=frame)

        ret, self.img = self.capture.read()
        self.displayImage(self.image, 2)


    def loadClicked(self):
        img, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image File (*.jpg)")
        if img:
            self.loadImage(img)
        else:
            pass

    def loadImage(self,img):

        self.image = cv2.imread(img, cv2.IMREAD_COLOR)
        self.displayImage(self.image, 2)

    def predict_number(self, img):
       pred = predict_single_image(class_names, model_ft, 'test.jpg')
       #if (pred == 'Malignant'):
      # self.PredictedResult.setText('We think you have Cancer,'
                                   ' Do not panic and see a dermatologist as soon as possible')
       #self.PredictedResult.setText('you have no cancer, congratulations')
     # else:
     #  self.PredictedResult.setText('you have no cancer, congratulations')
       #print(pred)
      self.PredictedResult.setText(str(pred))




        # pass
        # qformat = QImage.Format_RGB888
        # outImage = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # # outImage = outImage.rgbSwapped()
        # self.imgLabel3.setPixmap(QPixmap.fromImage(outImage))
        # self.imgLabel3.setScaledContents(True)



    def start_webcam(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret,self.image=self.capture.read()
        # self.image=cv2.flip(self.image,1)
        self.displayImage(self.image,1)

    def stop_webcam(self):
        self.timer.stop()


    def displayImage(self,img,window=1):
        qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        if window==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
        if window==2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel2.setScaledContents(True)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=GUI()
    window.setWindowTitle('Cancer Detection GUI')
    window.show()
    sys.exit(app.exec_())