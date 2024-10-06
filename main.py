# main.py

import logging
from ocr_processor import OCRProcessor
from key_listener import KeyListener
from llm_handler import LLMHandler

def main():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        model_path = "./models/llama-3.2-3b-instruct-q4_k_m.gguf"

        llm_handler = LLMHandler(model_path)

        ocr_processor = OCRProcessor()

        key_listener = KeyListener(ocr_processor, llm_handler)
        key_listener.start()
    except Exception as e:
        logger.exception(f"Error in main: {e}")

if __name__ == "__main__":
    main()
