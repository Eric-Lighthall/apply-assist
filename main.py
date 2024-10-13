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

import cv2
import os
import numpy as np
from datetime import datetime

class InputBoxProcessor:
    def __init__(self):
        self.ocr_processor = OCRProcessor(languages=['en'], use_gpu=True)
        self.llm_handler = LLMHandler()
        self.input_box_queue = deque()
        self.screenshot_counter = 0
        self.ocr_screenshot_dir = "./ocr_screenshots"
        self.debug_image_dir = "./debug_images"
        os.makedirs(self.ocr_screenshot_dir, exist_ok=True)
        os.makedirs(self.debug_image_dir, exist_ok=True)

    def detect_input_boxes(self, image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate the edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours on the dilated edge image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_image = image_cv.copy()
        print(f"Total contours found: {len(contours)}")
        
        for i, contour in enumerate(contours):
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the polygon has 4 vertices (rectangle) or is a long straight line
            if len(approx) == 4 or (len(approx) == 2 and cv2.arcLength(contour, True) > 200):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                
                # Adjusted criteria for input boxes: allow for very wide boxes
                if ((w > 100 and h > 20 and area > 3000 and 1.5 <= aspect_ratio <= 50) or
                    (w > 300 and 10 <= h <= 50)):  # Special case for very wide, short boxes
                    
                    # Check if the rectangle is relatively "clean" (not filled with text)
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    roi = cv2.bitwise_and(edges, mask)
                    if cv2.countNonZero(roi) < 0.15 * area:  # Relaxed threshold for wide boxes
                        # Additional check for very wide boxes: ensure it's not just a line
                        if h > 5 or (h <= 5 and cv2.countNonZero(roi) > 0):
                            self.input_box_queue.append((x, y, w, h))
                            cv2.drawContours(debug_image, [contour], 0, (0, 255, 0), 2)
                            print(f"Box {i}: x={x}, y={y}, w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}, area={area}")
                        else:
                            cv2.drawContours(debug_image, [contour], 0, (255, 255, 0), 1)  # Yellow for possible lines
                    else:
                        cv2.drawContours(debug_image, [contour], 0, (255, 0, 0), 1)  # Blue for ignored due to internal content
                else:
                    cv2.drawContours(debug_image, [contour], 0, (0, 0, 255), 1)  # Red for rejected
                    print(f"Rejected {i}: x={x}, y={y}, w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}, area={area}")
        
        print(f"Detected {len(self.input_box_queue)} input boxes")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_image_path = os.path.join(self.debug_image_dir, f"detected_boxes_{timestamp}.png")
        cv2.imwrite(debug_image_path, debug_image)
        print(f"Saved debug image with detected boxes: {debug_image_path}")
        
        # Save the original image and edge image for comparison
        original_image_path = os.path.join(self.debug_image_dir, f"original_image_{timestamp}.png")
        cv2.imwrite(original_image_path, image_cv)
        edge_image_path = os.path.join(self.debug_image_dir, f"edge_image_{timestamp}.png")
        cv2.imwrite(edge_image_path, edges)
        print(f"Saved original and edge images: {original_image_path}, {edge_image_path}")

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