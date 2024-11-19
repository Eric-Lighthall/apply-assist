import cv2
import numpy as np

def create_radio_template():
    # Create a black image
    size = 20  # or whatever size your radio buttons are
    template = np.zeros((size, size), dtype=np.uint8)
    
    # Draw a filled white circle
    center = (size // 2, size // 2)
    radius = size // 3
    cv2.circle(template, center, radius, 255, -1)  # -1 means filled
    
    # Save the template
    cv2.imwrite('templates/radio2.png', template)
    print("Template saved as radio2.png")

if __name__ == "__main__":
    create_radio_template()