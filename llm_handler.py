# llm_handler.py

import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, model_path):
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                verbose=False
            )
            logger.info("LLM model loaded successfully.")
        except Exception as e:
            logger.exception(f"Error loading LLM model: {e}")
            self.llm = None

    def process_text(self, prompt_text):
        if not self.llm:
            logger.error("LLM model is not loaded.")
            return "LLM model is not available."

        try:
            prompt = f"""
You are an assistant that helps in form-filling automation. Given the following text extracted from a form field label: '{prompt_text}', identify the type of data expected in the input field from the following options:

- first_name
- last_name
- email_address
- phone_number
- address
- city
- state
- zip_code
- country
- school_name
- start_date
- end_date
- major
- degree

Respond with only the field type in lowercase underscore format as above.
"""
            response = self.llm(
                prompt,
                max_tokens=50,
                stop=["\n"],
                echo=False,
                temperature=0.1
            )
            ai_response = response['choices'][0]['text'].strip()
            return ai_response
        except Exception as e:
            logger.exception(f"Error processing text with LLM: {e}")
            return "Error processing text with LLM."
