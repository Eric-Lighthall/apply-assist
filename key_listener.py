# key_listener.py

import logging
import threading
import cv2
import numpy as np
import pyautogui
from pynput import keyboard, mouse
import time
import difflib
import pyperclip

logger = logging.getLogger(__name__)

class KeyListener:
    def __init__(self, ocr_processor, llm_handler):
        self.ocr_processor = ocr_processor
        self.llm_handler = llm_handler
        self.control_pressed = False

    def on_press(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.control_pressed = True

    def on_release(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.control_pressed = False

    def on_click(self, x, y, button, pressed):
        if pressed and self.control_pressed and button == mouse.Button.left:
            threading.Thread(target=self.process_with_ocr, args=(x, y)).start()

    def process_with_ocr(self, x, y):
        logger.info(f"Processing click at ({x}, {y})")
        width = 300
        height = 50
        left = max(x - width // 2, 0)
        top = max(y - height - 10, 0)

        try:
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            full_text, formatted_results = self.ocr_processor.extract_text(screenshot)

            if full_text:
                print(f"\nExtracted text above click: {full_text}")
                logger.info(f"Extracted text above click: {full_text}")

                # Process the extracted text with the LLM
                llm_response = self.llm_handler.process_text(full_text)
                print(f"\nLLM Response:\n{llm_response}")
                logger.info(f"LLM Response: {llm_response}")

                # Map the LLM response to data
                data_mapping = {
                    'first_name': 'Eric',
                    'middle_name': 'Gordon',
                    'last_name': 'Lighthall',
                    'email_address': 'ericlighthall2@gmail.com',
                    'phone_number': '4256334518',
                    'address': '12914 NE 201st Way',
                    'city': 'Woodinville',
                    'state': 'WA',
                    'zip_code': '98072',
                    'country': 'USA',
                    'school_name': 'Lake Washington Institute of Technology',
                    'start_date': '01/08/2024',
                    'end_date': '12/07/2025',
                    'major': 'Computing and Software Development',
                    'degree': 'BAS',
                }

                field_type = llm_response.strip().lower()

                possible_fields = list(data_mapping.keys())
                closest_matches = difflib.get_close_matches(field_type, possible_fields, n=1, cutoff=0.6)

                if closest_matches:
                    matched_field = closest_matches[0]
                    data_to_input = data_mapping[matched_field]

                    pyautogui.click(x, y)
                    time.sleep(0.1)

                    pyperclip.copy(data_to_input)
                    pyautogui.hotkey('ctrl', 'v')

                    logger.info(f"Input data: {data_to_input} into field: {matched_field}")
                    print(f"Filled in '{matched_field}' with '{data_to_input}'.")
                else:
                    logger.warning(f"Field type '{field_type}' not recognized.")
                    print(f"Field type '{field_type}' not recognized.")
            else:
                print("No text detected.")
                logger.info("No text detected.")

        except Exception as e:
            logger.exception(f"Error in process_with_ocr: {e}")

    def start(self):
        keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        mouse_listener = mouse.Listener(on_click=self.on_click)
        keyboard_listener.start()
        mouse_listener.start()
        print("Listeners started. Press Ctrl+Click near an input box to perform OCR.")
        logger.info("Listeners started. Press Ctrl+Click on an input box.")
        keyboard_listener.join()
        mouse_listener.join()
