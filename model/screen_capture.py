import pyautogui
import numpy as np
import cv2
import time
from pathlib import Path
from screeninfo import get_monitors
from text_processor import OCRVisualizer
from form_detector import FormElementDetector

# Constants for offsets
TOP_OFFSET = 250
BOTTOM_OFFSET = 90

def get_monitor_with_mouse():
    """Get the monitor that contains the mouse cursor"""
    mouse_x, mouse_y = pyautogui.position()
    
    for monitor in get_monitors():
        if (monitor.x <= mouse_x < monitor.x + monitor.width and 
            monitor.y <= mouse_y < monitor.y + monitor.height):
            return monitor
    
    return None

def capture_monitor():
    """Capture the monitor that contains the mouse cursor, excluding top and bottom offsets"""
    monitor = get_monitor_with_mouse()
    if monitor is None:
        raise Exception("Could not determine which monitor contains the mouse cursor")
    
    # Calculate the actual capture region with offsets
    capture_height = monitor.height - (TOP_OFFSET + BOTTOM_OFFSET)
    
    # Capture the specific monitor region
    screenshot = pyautogui.screenshot(region=(
        monitor.x,
        monitor.y + TOP_OFFSET,  # Start after top offset
        monitor.width,
        capture_height  # Exclude bottom offset
    ))
    return np.array(screenshot)

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("Position your mouse on the monitor you want to capture...")
    print(f"Will capture excluding top {TOP_OFFSET}px and bottom {BOTTOM_OFFSET}px")
    print("Capturing in 1.5 seconds...")
    time.sleep(1.5)
    
    # Capture monitor
    image = capture_monitor()
    # Convert from RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save initial screenshot
    initial_path = str(output_dir / "screenshot.png")
    cv2.imwrite(initial_path, image)
    print(f"Screenshot saved to: {initial_path}")
    
    # Initialize OCR and process text
    visualizer = OCRVisualizer(use_gpu=False)
    masked_path = str(output_dir / "text_masked.jpg")
    text_masked_image = visualizer.process_image(initial_path, masked_path)
    print(f"Text masked image saved to: {masked_path}")
    
    # Detect form controls
    detector = FormElementDetector(use_gpu=False)
    controls_path = str(output_dir / "form_controls.jpg")
    result, elements = detector.detect_shapes(masked_path, initial_path, controls_path)
    print(f"Form controls detected and saved to: {controls_path}")
    
    print("\nProcessing complete! Check the output directory for results:")
    print(f"1. Original screenshot: {initial_path}")
    print(f"2. Text masked image: {masked_path}")
    print(f"3. Form controls detected: {controls_path}")

if __name__ == "__main__":
    main()