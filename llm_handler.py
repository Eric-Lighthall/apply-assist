import logging
import re
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LlamaClassifier:
    def __init__(
        self,
        model_path="./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1  # Adjust based on available GPU ram. -1 = offload all ram to GPU, low values offload more, high offloads less.
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=32,
            n_gpu_layers=n_gpu_layers
        )
        self.categories = [
            "first_name", "middle_name", "last_name", "email_address",
            "phone_number", "address", "city", "state", "zip_code",
            "country", "school_name", "start_date", "end_date", "major", "degree",
            "job_title", "role_description", "company", "location", "gpa",
            "field_of_study", "from", "to_actual_expected", "password"
        ]

    def classify(self, text):
        prompt = f"""Classify the following text into one of these categories:
{', '.join(self.categories)}
If you see something like Number, Phone, Phone number, that means it's phone_number.
Location is location.
If you see Address 2, it's none of the categories. If Address or Address line or Address line 1 appears and there's no 2, then it is the category address.
Pick the closest related option, and only respond with the one-word classification.
Text: "{text}"
Classification:"""

        response = self.llm(
            prompt,
            max_tokens=20,
            stop=["\n"],
            echo=False,
            temperature=0.2
        )
        
        raw_classification = response['choices'][0]['text']
        print(f"Raw classification: '{raw_classification}'")
        
        classification = raw_classification.strip().lower()
        classification = re.sub(r'\s+', '', classification)  # Remove all whitespace
        print(f"Processed classification: '{classification}'")
        
        if classification.lower() not in [cat.lower() for cat in self.categories]:
            print(f"Classification '{classification}' not in categories: {self.categories}")
            return "unknown"
        
        print(f"Classification is {classification}")
        return classification

class LLMHandler:
    def __init__(self):
        self.classifier = LlamaClassifier()
        logger.info("LLMHandler initialized with Llama model for classification.")

    def process_text(self, prompt_text):
        try:
            field_type = self.classifier.classify(prompt_text)
            logger.info(f"Classified '{prompt_text}' as '{field_type}'")
            return field_type
        except Exception as e:
            logger.exception(f"Error processing text: {e}")
            return "error"

# Usage
# handler = LLMHandler()
# Use handler.process_text(input_text) to classify text