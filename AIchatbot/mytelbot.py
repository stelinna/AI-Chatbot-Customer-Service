import os
import time
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from logging_config import logger
from sentence_transformers import SentenceTransformer
from conversation_utils import get_faq_match
from conversation_manager import ConversationManager
from memory_utils import save_message, retrieve_relevant_memory
from intent_utils import detect_thank_you, detect_exit
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
# ---------------------------
# Load API key
# ---------------------------
load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# ---------------------------
# Load FAQ data
# ---------------------------
def read_data_from_file(filename="data.yaml"):
    with open(filename, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

support_data = read_data_from_file("data.yaml")

# ---------------------------
# FAQ embeddings
# ---------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

faq_questions = []
faq_answers = []
faq_embeddings = []

for qa in support_data.get("questions_and_answers", []):
    question = qa.get("question", "").strip()
    answer = " ".join(qa.get("answer", [])).strip()

    faq_questions.append(question)
    faq_answers.append(answer)
    faq_embeddings.append(
        embed_model.encode(question, show_progress_bar=False))

# ---------------------------
# API response 
# ---------------------------
def generate_response(query, last_turns):
    context = "\n".join(last_turns[-2:]) if last_turns else ""

    prompt = (
        f"FAQ DATA:\n{yaml.dump(support_data)}\n\n"
        f"Conversation context:\n{context}\n\n"
        f"User question: {query}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                      "content": (
                            "You are a customer support assistant. "
                            "Provide clear, complete, and helpful answers. "
                            "Include all relevant information from the FAQ or context. "
                            "Do NOT repeat or rephrase the user's question. Start directly with the answer. "
                            "Keep answers professional and easy to understand, concise but informative.\n"
                            "Rules you MUST follow:\n"
                            "- Be polite, respectful, and neutral.\n"
                            "- NEVER produce bullying, harassment, hate speech, threats, or offensive language.\n"
                            "- NEVER insult, mock, or judge the user.\n"
                            "- Answer ONLY using the provided FAQ data.\n"
                            "- If the question is inappropriate or unsafe, respond with a calm refusal.\n"
                      )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=180
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.log_message("Error", str(e))
        return "Something went wrong. Please try again later."

# ---------------------------
def get_auto_response(user_input, last_turns, conv_manager, logger):
    import re

    query_clean = re.sub(r"[^\w\s]", "", user_input.lower().strip())

    cached = conv_manager.get_cached_response(query_clean)
    if cached:
        logger.log_message("System", "Response source: CONVERSATION CACHE")
        return cached

    thank = detect_thank_you(user_input)
    if thank:
        logger.log_message("System", "Response source: THANK YOU INTENT")
        return thank

    for i, q in enumerate(faq_questions):
        question_clean = re.sub(r"[^\w\s]", "", q.lower())
        if query_clean == question_clean:
            answer = faq_answers[i]
            conv_manager.save_to_cache(user_input, answer)
            logger.log_message("System", "Response source: EXACT FAQ")
            return answer

    faq_index = get_faq_match(
        query_clean,
        [re.sub(r"[^\w\s]", "", q.lower()) for q in faq_questions],
        faq_embeddings,
        threshold=0.60  
    )

    if faq_index is not None:
        answer = faq_answers[faq_index]
        conv_manager.save_to_cache(user_input, answer)
        logger.log_message("System", "Response source: SEMANTIC FAQ")
        return answer

    last_bot_answer = ""
    for turn in reversed(last_turns):
        if turn.startswith("Bot:"):
            last_bot_answer = turn.replace("Bot:", "").strip()
            break

    if last_bot_answer:
        followup_query = f"{last_bot_answer} {user_input}"
        followup_clean = re.sub(r"[^\w\s]", "", followup_query.lower().strip())

        faq_index = get_faq_match(
            followup_clean,
            [re.sub(r"[^\w\s]", "", q.lower()) for q in faq_questions],
            faq_embeddings,
            threshold=0.60
        )

        if faq_index is not None:
            answer = faq_answers[faq_index]
            conv_manager.save_to_cache(user_input, answer)
            logger.log_message("System", "Response source: FOLLOW-UP FAQ")
            return answer

    logger.log_message("System", "Response source: OUT OF SCOPE")
    return (
        "I'm sorry, this question is outside my area of support. "
        "I can help with products, orders, shipping, returns, and promotions."
    )

# ---------------------------
# MAIN LOOP
# ---------------------------
def main():
    print("Welcome to customer support!")
    print("How may I help you?")

    conversation_history = []
    conv_manager = ConversationManager()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if detect_exit(user_input):
            print("Bot: Goodbye! Have a great day! 👋")
            break

        save_message(user_input, role="user")
        conversation_history.append(f"You: {user_input}")

        last_turns = conversation_history[-6:]

        answer = get_auto_response(user_input, last_turns, conv_manager, logger)
        logger.log_message("User", user_input)
        save_message(user_input, role="user")
        conversation_history.append(f"You: {user_input}")
        logger.log_message("Bot", answer)
        print("Bot:", answer)
        save_message(answer, role="bot", linked_question=user_input)

# ---------------------------
if __name__ == "__main__":
    main()
