import sys
import cv2
import numpy as np
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from collections import deque
from ocr_processor import OCRProcessor
from llm_handler import LLMHandler
import time
import os
import pyperclip
from dotenv import load_dotenv

load_dotenv()

class InputBoxProcessor:
    def __init__(self):
        self.ocr_processor = OCRProcessor(languages=['en'], use_gpu=True)
        self.llm_handler = LLMHandler()
        self.input_box_queue = deque()
        self.screenshot_counter = 0
        self.ocr_screenshot_dir = "./ocr_screenshots"
        os.makedirs(self.ocr_screenshot_dir, exist_ok=True)

    def detect_input_boxes(self, image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.9 <= aspect_ratio <= 10 and w > 50 and h > 20 and area > 1000:
                self.input_box_queue.append((x, y, w, h))
                cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite("./results/result.png", image_cv)

    def process_input_boxes(self, base_x, base_y):
        while self.input_box_queue:
            self.process_single_box(self.input_box_queue.popleft(), base_x, base_y)

    def process_single_box(self, box, base_x, base_y):
        x, y, w, h = box
        pyautogui.moveTo(base_x + x + w/2, base_y + y + h/2)
        
        ocr_height = 30
        ocr_area = pyautogui.screenshot(region=(base_x + x, base_y + y - ocr_height, w, ocr_height))
        ocr_image = cv2.cvtColor(np.array(ocr_area), cv2.COLOR_RGB2BGR)
        
        self.save_ocr_screenshot(ocr_image)
        
        text, _ = self.ocr_processor.extract_text(ocr_image)
        input_type = self.llm_handler.process_text(text)
        input_value = self.generate_input(input_type)

        pyautogui.click()
        time.sleep(0.02)
        pyautogui.click()
        pyautogui.click()
        time.sleep(0.02)
        pyperclip.copy(input_value)
        pyautogui.hotkey('ctrl', 'v')

    def save_ocr_screenshot(self, ocr_image):
        self.screenshot_counter += 1
        filename = f"ocr_area_{self.screenshot_counter}.png"
        filepath = os.path.join(self.ocr_screenshot_dir, filename)
        cv2.imwrite(filepath, ocr_image)

    def generate_input(self, input_type):
        input_map = {
            'first_name': os.getenv('FIRST_NAME'),
            'middle_name': os.getenv('MIDDLE_NAME'),
            'last_name': os.getenv('LAST_NAME'),
            'email_address': os.getenv('EMAIL_ADDRESS'),
            'phone_number': os.getenv('PHONE_NUMBER'),
            'address': os.getenv('ADDRESS'),
            'city': os.getenv('CITY'),
            'state': os.getenv('STATE'),
            'zip_code': os.getenv('ZIP_CODE'),
            'country': os.getenv('COUNTRY'),
            'school_name': os.getenv('SCHOOL_NAME'),
            'start_date': os.getenv('START_DATE'),
            'end_date': os.getenv('END_DATE'),
            'major': os.getenv('MAJOR'),
            'degree': os.getenv('DEGREE'),
            'job_title': os.getenv('JOB_TITLE'),
            'company': os.getenv('COMPANY'),
            'location': os.getenv('LOCATION'),
            'gpa': os.getenv('GPA'),
            'field_of_study': os.getenv('FIELD_OF_STUDY'),
            'from': os.getenv('FROM_YEAR'),
            'to_actual_expected': os.getenv('TO_YEAR'),
            'role_description': os.getenv('ROLE_DESCRIPTION'),
            'password': os.getenv('PASSWORD')
        }
        return input_map.get(input_type, '')

class SnippingWidget(QtWidgets.QWidget):
    def __init__(self, screenshot, screen_geometry, processor):
        super().__init__()
        self.screenshot = screenshot
        self.screen_geometry = screen_geometry
        self.processor = processor
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setGeometry(screen_geometry)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.start_point = QtCore.QPoint()
        self.end_point = QtCore.QPoint()
        self.selection_rect = QtCore.QRect()
        self.setCursor(QtCore.Qt.CrossCursor)
        self.selection_complete = False

    def paintEvent(self, event):
        if self.selection_complete:
            return
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 0))
        if not self.selection_rect.isNull():
            pen = QtGui.QPen(QtGui.QColor('red'), 2)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect.normalized())
        painter.end()

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.selection_rect = QtCore.QRect(self.start_point, QtCore.QSize())
        self.update()

    def mouseMoveEvent(self, event):
        self.selection_rect = QtCore.QRect(self.start_point, event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.selection_complete = True
        self.hide()
        self.capture_snip()

    def capture_snip(self):
        x1, y1 = min(self.start_point.x(), self.end_point.x()), min(self.start_point.y(), self.end_point.y())
        x2, y2 = max(self.start_point.x(), self.end_point.x()), max(self.start_point.y(), self.end_point.y())
        cropped_image = self.screenshot.crop((x1, y1, x2, y2))
        self.processor.detect_input_boxes(cropped_image)
        self.processor.process_input_boxes(x1, y1)
        QtWidgets.QApplication.quit()

def get_primary_screen():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    return next((screen for screen in app.screens() if screen.geometry().x() == 0 and screen.geometry().y() == 0), app.primaryScreen())

def main():
    app = QtWidgets.QApplication(sys.argv)
    primary_screen = get_primary_screen()
    screen_geometry = primary_screen.geometry()
    
    screenshot = pyautogui.screenshot(region=(
        screen_geometry.x(),
        screen_geometry.y(),
        screen_geometry.width(),
        screen_geometry.height()))
    
    processor = InputBoxProcessor()
    snip = SnippingWidget(screenshot, screen_geometry, processor)
    snip.showFullScreen()
    snip.activateWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()