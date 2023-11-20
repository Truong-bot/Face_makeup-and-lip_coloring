import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)

import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit
from PyQt5.QtGui import QPixmap
import os

from UI_Maker_Face import Ui_MainWindow
import cv2
import makeup

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.pushButton_3.clicked.connect(self.show_image_orig)
        self.pushButton.clicked.connect(self.run)
        self.pushButton_5.clicked.connect(self.show_image_targ)
        self.pushButton_2.clicked.connect(self.save_file)
        self.pushButton_4.clicked.connect(self.exit)
        self.arr_orig = []
        self.arr_targ = []
        self.arr_img = []


    def exit(self):
        sys.exit()

    def save_file(self):
        fileName = QFileDialog.getSaveFileName(self, str("Save File"), "/Users/hatk/Desktop/FaceMakeUpProject/result/untitled.jpg", str("Images (*.png *.xpm *.jpg)"))
        fileName = fileName[0]
        if fileName != None:
            fileName = list(fileName.split('/'))
            fileName = str(fileName[len(fileName) - 2]) + '/' + str(fileName[len(fileName)-1])
            img = self.arr_img[len(self.arr_img) - 1]
            cv2.imwrite(fileName, img)


    def run(self):
        self.read_import()

    def read_import(self):
        url_orig = self.arr_orig[len(self.arr_orig)-1]
        url_orig = list(url_orig.split('/'))
        url_orig = str(url_orig[len(url_orig)-2]) + '/' + str(url_orig[len(url_orig)-1])

        url_targ = self.arr_targ[len(self.arr_targ) - 1]
        url_targ = list(url_targ.split('/'))
        url_targ = str(url_targ[len(url_targ) - 2]) + '/' + str(url_targ[len(url_targ) - 1])

        img = makeup.makeup_main(url_orig, url_targ)

        img = self.crop_img(img)

        if len(self.arr_img) != 0:
            self.arr_img.pop(0)
        self.arr_img.append(img)

        scene = QtWidgets.QGraphicsScene(self.graphicsView_2)
        pixmap = self.convert_cv_qt(img)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView_2.setScene(scene)

    def crop_img(self, img):
        w, h = img.shape[:2]
        for i in range(0, h - 1)[::-1]:
            if img[i][0][0] > 10:
                img = img[0:i, :]
                break
        return img

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = PyQt5.QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaledToHeight(570)
        return QPixmap.fromImage(p)

    def open_dialog_box_orig(self):
        filename = QFileDialog.getOpenFileName(self, str("Open File"), "/Users/hatk/Desktop/FaceMakeUpProject/input_image/", str("Images (*.png *.xpm *.jpg *.jpeg)"))
        path = filename[0]
        if len(self.arr_orig) != 0:
            self.arr_orig.pop(0)
        self.arr_orig.append(path)
        return path

    def open_dialog_box_targ(self):
        filename = QFileDialog.getOpenFileName(self, str("Open File"), "/Users/hatk/Desktop/FaceMakeUpProject/target_image/", str("Images (*.png *.xpm *.jpg *.jpeg)"))
        path = filename[0]
        if len(self.arr_targ) != 0:
            self.arr_targ.pop(0)
        self.arr_targ.append(path)
        return path


    def show_image_orig(self):
        image_path = self.open_dialog_box_orig()
        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self.graphicsView)
            pixmap = QPixmap(image_path).scaledToHeight(570)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView.setScene(scene)

    def show_image_targ(self):
        image_path = self.open_dialog_box_targ()
        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self.graphicsView_3)
            pixmap = QPixmap(image_path).scaledToHeight(570)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView_3.setScene(scene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())


    # myself
    # subject = cv2.imread('../input_image/IMG_1833.jpeg', 1)
    # subject = cv2.imread('../input_image/test_image_2.jpg', 1)
    # target = cv2.imread('../target_image/man_with_make_up.jpg', 1)
    # target = cv2.imread('../target_image/black_man.jpeg', 1)
    # subject = cv2.imread('../input_image/test_image_2.jpg', 1)
    # target = cv2.imread('../target_image/target_11.jpg', 1)