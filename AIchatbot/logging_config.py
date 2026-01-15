import logging
import tiktoken

class Logger:
    def __init__(self, filename, print_to_console=False): 
        """
        Logger for messages and token counts.
        """
        self.filename = filename
        self.print_to_console = print_to_console
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

    def log_message(self, sender, message):
        context_tokens = len(self.encoding.encode(message))
        log_text = f'{sender}: {message}\nContext tokens: {context_tokens}'
        logging.info(log_text)
        if self.print_to_console:
            print(log_text)

    def log_generated_tokens(self, generated_tokens):
        log_text = f'Generated tokens: {generated_tokens}'
        logging.info(log_text)
        if self.print_to_console:
            print(log_text)

logger = Logger('conversation.log', print_to_console=False)

