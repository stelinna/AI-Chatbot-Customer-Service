from conversation_utils import get_cached_answer, add_to_cache
from memory_utils import get_memory_cached_answer
from logging_config import logger

class ConversationManager:
    def __init__(self):
        self.session_history = []

    def add_turn(self, user, bot):
        self.session_history.append({
            "user": user,
            "bot": bot
        })

    def get_cached_response(self, user_input):
        cached = get_cached_answer(user_input)
        if cached:
            logger.log_message(
                "System",
                "Response source: YAML CACHE (API skipped)"
            )
            return cached

        memory_cached = get_memory_cached_answer(user_input)
        if memory_cached:
            logger.log_message(
                "System",
                "Response source: MEMORY CACHE (API skipped)"
            )
            return memory_cached

        return None

    def save_to_cache(self, user_input, bot_answer):
        add_to_cache(user_input, bot_answer)
