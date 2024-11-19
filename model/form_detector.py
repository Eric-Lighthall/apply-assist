import cv2
import numpy as np
from queue import Queue
from paddleocr import PaddleOCR

class FormElementDetector:
    def __init__(self, use_gpu=False):
        self.reader = PaddleOCR(
            lang='en',
            use_angle_cls=False,
            use_gpu=use_gpu,
            det_limit_type='max',
            debug=False,
            show_log=False,
        )

    def get_element_label(self, original_image, x1, y1, x2, y2, element_type):
        """Extract and recognize text above or to the right of a form element"""
        if element_type == 'checkbox':
            # For checkboxes, look to the right
            label_width = 300  # Width of scan area
            label_x1 = x2  # Start from right edge of checkbox
            label_x2 = min(original_image.shape[1], label_x1 + label_width)
            label_y1 = max(0, y1 - 10)  # Small vertical padding
            label_y2 = min(original_image.shape[0], y2 + 10)
        else:
            # For other elements, look above
            label_height = 60
            label_y1 = max(0, y1 - label_height)
            label_y2 = y1
            padding = 50
            label_x1 = max(0, x1 - padding)
            label_x2 = min(original_image.shape[1], x2 + padding)
        
        # Extract the region from the original image
        label_region = original_image[label_y1:label_y2, label_x1:label_x2]
        
        # Draw rectangle around OCR area (for debugging)
        cv2.rectangle(original_image, (label_x1, label_y1), (label_x2, label_y2), (255, 0, 255), 2)
        cv2.putText(original_image, f"OCR area", (label_x1, label_y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imwrite('output/ocr_debug.jpg', original_image)
        
        # Perform OCR on the region
        result = self.reader.ocr(label_region, cls=True)
        
        if result is not None and len(result) > 0:
            texts = []
            for line in result:
                if line:
                    for detection in line:
                        text = detection[1][0]
                        confidence = detection[1][1]
                        if confidence > 0.5:
                            texts.append(text)
            
            return ' '.join(texts)
        return None

    def detect_shapes(self, image_path, original_image_path, output_path):
        """Detect form controls using template matching and extract their labels."""
        # Read the masked image for form detection
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the masked image")
            
        # Read the original image for OCR
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            raise ValueError("Could not read the original image")
        
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Colors for different controls
        COLORS = {
            'checkbox': (255, 0, 0),     # Blue
            'radio': (0, 255, 0),        # Green
            'textinput': (255, 165, 0),  # Orange
            'dropdown': (128, 0, 128)    # Purple
        }
        
        templates = {
            'checkbox': {
                'path': 'templates/checkbox.png',
                'threshold': 0.8,
                'color': COLORS['checkbox'],
                'nms_threshold': 0.3
            },
            'radio': {
                'path': 'templates/radio.png',
                'threshold': 0.8,
                'color': COLORS['radio'],
                'nms_threshold': 0.3
            },
            'textinput': {
                'path': 'templates/textinput.png',
                'threshold': 0.75,
                'color': COLORS['textinput'],
                'nms_threshold': 0.5
            }
        }
        
        detected_elements = []
        
        # Process each template
        for control_type, props in templates.items():
            try:
                template = cv2.imread(props['path'], cv2.IMREAD_GRAYSCALE)
                if template is None:
                    print(f"Could not read template: {props['path']}")
                    continue
                    
                template_h, template_w = template.shape
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                
                locations = np.where(result >= props['threshold'])
                scores = result[locations[0], locations[1]]
                coordinates = list(zip(*locations[::-1]))
                
                if not coordinates:
                    continue
                    
                boxes = []
                for (x, y) in coordinates:
                    boxes.append([x, y, x + template_w, y + template_h])
                
                keep_indices = non_max_suppression(boxes, scores, props['nms_threshold'])
                
                for idx in keep_indices:
                    x1, y1, x2, y2 = boxes[idx]
                    
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.shape[1], x2 + padding)
                    y2 = min(image.shape[0], y2 + padding)
                    
                    current_type = control_type
                    current_color = props['color']
                    
                    if control_type == 'textinput':
                        # Draw the right box
                        box_coords = draw_right_box(output, x2, y1, y2, current_color)
                        
                        # Check for grey in the box
                        if check_for_grey(image, box_coords):
                            current_type = 'dropdown'
                            current_color = COLORS['dropdown']
                    
                    # Get the label text for this element from the original image
                    label_text = self.get_element_label(original_image, x1, y1, x2, y2, current_type)
                    
                    # Store detection info
                    detected_elements.append({
                        'type': current_type,
                        'label': label_text,
                        'coords': (x1, y1, x2, y2)
                    })
                    
                    # Draw control outline with appropriate color
                    cv2.rectangle(output, (x1, y1), (x2, y2), current_color, 2)
                    
                    # Print detection info
                    print(f"\nDetected {current_type.upper()}:")
                    print(f"Label text: {label_text if label_text else 'No label found'}")
                    print(f"Position: ({x1}, {y1})")
                    
                    # Add visual label
                    cv2.putText(output, f"{current_type}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
                    
                    # If it's a dropdown, redraw the right box with dropdown color
                    if current_type == 'dropdown':
                        draw_right_box(output, x2, y1, y2, current_color)
                    
            except Exception as e:
                print(f"Error processing {control_type}: {str(e)}")
                continue
        
        cv2.imwrite(output_path, output)
        return output, detected_elements

# Keep existing helper functions unchanged
def non_max_suppression(boxes, scores, threshold):
    """Apply non-maxima suppression to avoid multiple detections"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        if indices.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[indices[1:]]
        
        indices = indices[1:][overlap < threshold]
    
    return keep

def check_for_grey(image, box_coords):
    """Check if grey pixels exist in the box region"""
    x1, y1, x2, y2 = box_coords
    roi = image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lower_grey = 128
    upper_grey = 192
    grey_pixels = np.sum((gray_roi >= lower_grey) & (gray_roi <= upper_grey))
    return grey_pixels > 50

def draw_right_box(image, x2, y1, y2, color):
    """Draw a 30x30 box near the right edge of a text input"""
    box_size = 30
    margin = 20
    box_x1 = x2 - box_size - margin
    box_y1 = y1 + (y2 - y1 - box_size) // 2
    box_x2 = x2 - margin
    box_y2 = box_y1 + box_size
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, 2)
    return box_x1, box_y1, box_x2, box_y2

if __name__ == "__main__":
    try:
        detector = FormElementDetector(use_gpu=False)
        input_path = "text_masked_output.jpg"
        original_path = "screenshot.png"
        output_path = "output_image.jpg"
        
        result, elements = detector.detect_shapes(input_path, original_path, output_path)
        print(f"\nProcessing complete. Output saved to {output_path}")
        
        # Print summary of all detected elements
        print("\nSummary of detected elements:")
        for elem in elements:
            print(f"Type: {elem['type']}, Label: {elem['label']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")