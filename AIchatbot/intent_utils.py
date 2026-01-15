import re
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

THANK_WORDS = [
    "thank you",
    "thanks",
    "thx",
    "appreciate",
    "much appreciated",
    "bye",
    "goodbye"
]
EXIT_WORDS = [
    "bye",
    "goodbye",
    "see you",
    "exit",
    "quit"
]
THANK_RESPONSE = (
    "You're very welcome! 😊\n"
)

def detect_thank_you(text):
    """
    Detects a thank-you phrase and returns a polite response
    without ending the conversation.
    """
    text = text.lower().strip()
    for phrase in THANK_WORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return THANK_RESPONSE
    return None
def detect_exit(text):
    """
    Detects if the user wants to exit the chat.
    """
    text = text.lower().strip()
    for word in EXIT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return True
    return False