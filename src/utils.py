import os
import logging
from dotenv import load_dotenv

# დააყენეთ ლოგირება
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGChatbot")

def load_environment_variables() -> None:
    """ტვირთავს გარემოს ცვლადებს .env ფაილიდან."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY არ არის დაყენებული .env ფაილში.")
        # Streamlit-ის შემთხვევაში, გასაღები შეიძლება პირდაპირ UI-ში შეიყვანოს მომხმარებელმა.
        # თუმცა, საუკეთესო პრაქტიკაა მისი გარემოში შენახვა.

def get_gemini_api_key() -> str:
    """აბრუნებს Gemini API გასაღებს."""
    return os.getenv("GEMINI_API_KEY", "")

def log_info(message: str) -> None:
    """ინფორმაციის დალოგვა."""
    logger.info(message)