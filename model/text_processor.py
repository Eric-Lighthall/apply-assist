import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging

logger = logging.getLogger(__name__)

class OCRVisualizer:
    def __init__(self, languages=['en'], use_gpu=True):
        lang = '+'.join(languages)
        self.reader = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            lang_detect=True,
            use_gpu=use_gpu,
            det_limit_type='max',
            use_dilation=False
        )

    def detect_checkboxes(self, image):
        """
        Detect checkboxes using template matching
        """
        # Read the template
        template = cv2.imread('templates/checkbox.png', cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError("Could not read checkbox template")
            
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for checkboxes
        checkbox_mask = np.zeros_like(gray)
        
        # Get template dimensions
        h, w = template.shape
        
        # Template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        
        # Debug image
        debug_image = image.copy()
        
        # Find locations where template matching exceeds threshold
        locations = np.where(result >= threshold)
        
        # Create list of detection boxes
        boxes = []
        scores = []
        for pt in zip(*locations[::-1]):  # Switch columns and rows
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
            scores.append(result[pt[1], pt[0]])
            
        # Apply non-max suppression
        boxes = np.array(boxes)
        if len(boxes) > 0:
            # Convert to format expected by NMS
            scores = np.array(scores)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores, threshold, 0.3)
            
            if len(indices) > 0:
                # After NMS, draw remaining boxes
                for idx in indices:
                    if isinstance(idx, list):  # Handle different OpenCV versions
                        idx = idx[0]
                    x1, y1, x2, y2 = boxes[idx]
                    
                    # Add padding
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.shape[1], x2 + padding)
                    y2 = min(image.shape[0], y2 + padding)
                    
                    # Draw on mask
                    cv2.rectangle(checkbox_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
                    
                    # Draw on debug image (green rectangles)
                    cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    score = scores[idx]
                    cv2.putText(debug_image, f"score: {score:.2f}", 
                              (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
        
        return checkbox_mask

    def get_axis_aligned_box(self, points):
        """Convert any polygon to an axis-aligned rectangle"""
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        return np.array([
            [left, top + 1],
            [right, top + 1],
            [right, bottom - 1],
            [left, bottom - 1]
        ], dtype=np.int32)

    def should_keep_text(self, text):
        """Check if text should be kept (e.g., radio button options)"""
        text = text.lower().strip()
        radio_patterns = ['o yes', 'o no', '0 yes', '0 no', 'yes', 'no']
        return text in radio_patterns

    def process_image(self, image_path, masked_output_path):
        """
        Detect text and mask it with white in the image, preserving checkboxes.
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Create output copies
        masked_output = image.copy()
        debug_output = image.copy()
        
        # First detect checkboxes using template matching
        checkbox_mask = self.detect_checkboxes(image)

        # Perform OCR
        result = self.reader.ocr(image, cls=True)

        # Process detected regions
        if result is not None:
            for line in result:
                if line:
                    for detection in line:
                        box = detection[0]  # Coordinates
                        text = detection[1][0]  # Detected text
                        confidence = detection[1][1]  # Confidence score

                        # Print detection info
                        print(f"Detected: '{text}' (confidence: {confidence:.2f})")

                        # Skip if confidence is below threshold
                        if confidence < 0.8:
                            continue
                            
                        # Skip if should keep text
                        if self.should_keep_text(text):
                            continue

                        # Convert to numpy array and get axis-aligned box
                        points = np.array(box, dtype=np.int32)
                        aligned_box = self.get_axis_aligned_box(points)
                        
                        # Create a mask for this text region
                        text_mask = np.zeros_like(checkbox_mask)
                        cv2.fillPoly(text_mask, [aligned_box], 255)
                        
                        # Draw on debug image before checking overlap
                        cv2.polylines(debug_output, [points], True, (0, 255, 0), 2)
                        x = int(points[0][0])
                        y = int(points[0][1]) - 5
                        cv2.putText(debug_output, f"{text} ({confidence:.2f})", 
                                  (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)

                        # Check if this region overlaps with a checkbox
                        overlap = cv2.bitwise_and(checkbox_mask, text_mask)
                        if np.sum(overlap) > 0:
                            # If there's overlap, try to exclude the checkbox area
                            # by shifting the masking region right
                            aligned_box[:, 0] += 30  # Shift right by 30 pixels
                        
                        # Fill polygon with white
                        cv2.fillPoly(masked_output, [aligned_box], (255, 255, 255))

        # Save debug visualization
        debug_path = masked_output_path.replace('.jpg', '_debug.jpg')
        cv2.imwrite(debug_path, debug_output)
        logger.info(f"Debug visualization saved to {debug_path}")

        # Save masked result
        cv2.imwrite(masked_output_path, masked_output)
        logger.info(f"Masked image saved to {masked_output_path}")
        
        return masked_output, debug_output

# Example usage
if __name__ == "__main__":
    visualizer = OCRVisualizer(use_gpu=False)
    
    try:
        input_path = "input_image.jpg"
        output_path = "text_masked_output.jpg"
        
        masked_image, debug_image = visualizer.process_image(input_path, output_path)
        print(f"Processing complete!")
        print(f"Masked image saved to: {output_path}")
        print(f"Debug image saved to: {output_path.replace('.jpg', '_debug.jpg')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")