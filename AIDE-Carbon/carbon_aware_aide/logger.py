import sys
import logging
from io import StringIO
import contextlib

def log_llm_output( title, content):
    print("\n" + "=" * 60)
    print(f"[LLM OUTPUT] {title}")
    print("-" * 60)
    print(content)
    print("=" * 60)


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.log = logging.getLogger('run_all_logger')
        self.log.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        for handler in self.log.handlers[:]:
            self.log.removeHandler(handler)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.log.addHandler(file_handler)
        self.log.addHandler(console_handler)
    
    def info(self, message):
        self.log.info(message)
    
    def error(self, message):
        self.log.error(message)

@contextlib.contextmanager
def redirect_stdout_stderr(logger):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    class CustomStream:
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.buffer = StringIO()
        
        def write(self, message):
            self.buffer.write(message)
            if '\n' in message:
                self.flush()
        
        def flush(self):
            content = self.buffer.getvalue().strip()
            if content:
                self.logger.log(self.log_level, content)
                self.buffer = StringIO()
    
    sys.stdout = CustomStream(logger, logging.INFO)
    sys.stderr = CustomStream(logger, logging.ERROR)
    
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
