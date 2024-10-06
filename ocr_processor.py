# ocr_processor.py

import cv2
import numpy as np
import logging
import os
from datetime import datetime
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, languages=['en'], use_gpu=True):
        lang = '+'.join(languages)
        self.reader = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            lang_detect=True,
            use_gpu=use_gpu,
            det_db_thresh=0.3,
            det_db_box_thresh=0.3,
            rec_model_dir='en_PP-OCRv4_rec_infer',
            det_limit_side_len=2880,
            det_limit_type='max',
        )

    def extract_text(self, image):
        try:
            ocr_result = self.reader.ocr(image, rec=True, cls=True)
            logger.debug(f"OCR Result: {ocr_result}")

            extracted_text = []
            formatted_results = []

            if ocr_result and isinstance(ocr_result[0], list):
                ocr_result = ocr_result[0]

            for line in ocr_result:
                if len(line) >= 2:
                    bbox, (text_detected, prob) = line[0], line[1]
                    extracted_text.append(text_detected)
                    formatted_results.append((bbox, text_detected, prob))

            full_text = ' '.join(extracted_text)
            logger.debug(f"Extracted text: {full_text}")

            self.save_image_with_boxes(image, formatted_results)

            return full_text, formatted_results

        except Exception as e:
            logger.exception(f"Error in extract_text: {e}")
            return "", []

    def save_image_with_boxes(self, image, results):
        img_with_boxes = image.copy()

        for (bbox, text, prob) in results:
            bbox = np.array(bbox).astype(int).reshape(-1, 2)
            cv2.polylines(img_with_boxes, [bbox], True, (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{text} ({prob:.2f})", (bbox[0][0], bbox[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if not os.path.exists('ocr_results'):
            os.makedirs('ocr_results')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_results/ocr_result_{timestamp}.png"
        cv2.imwrite(filename, img_with_boxes)
        logger.info(f"Saved image with bounding boxes: {filename}")
