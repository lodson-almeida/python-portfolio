import random
import time
from bot.common.LA_log_wrapper import LogWrapper

class RetryMechanism:
    def __init__(self, base_wait=5, max_wait=200, max_failures=5, name="RetryMechanism", stop_on_max_failures=False):
        self.base_wait = base_wait
        self.max_wait = max_wait
        self.max_failures = max_failures
        self.attempt = 0
        self.name = name
        self.log = LogWrapper(name)
        self.stop_on_max_failures = stop_on_max_failures
        
    def reset(self):
        self.attempt = 0

    def before_retry(self):
        self.attempt += 1
        if self.attempt > self.max_failures:
            if self.stop_on_max_failures:
                self.log.logger.error(f"[{self.name}] Max failures ({self.max_failures}) reached. Stopping retries.")
                return False  # Indicate that we should stop retrying
            else:
                self.log.logger.debug(f"[{self.name}] Max failures reached, resetting attempt counter to 1")
                self.attempt = 1
        
        wait_time = min(self.base_wait * (2 ** (self.attempt - 1)), self.max_wait)
        jitter = random.uniform(0, wait_time / 2)
        total_wait = wait_time + jitter
        
        self.log.logger.error(f"[{self.name}] [Retry {self.attempt}] Connection failed. Retrying in {total_wait:.1f}s...")
        time.sleep(total_wait)
        self.log.logger.debug(f"[{self.name}] Attempting to reconnect... (attempt {self.attempt})")
        return True # Indicate that we should continue retrying

    def execute_with_retry(self, operation, *args, **kwargs):
        while True:
            try:
                result = operation(*args, **kwargs)
                self.reset()
                return result
            except Exception as e:
                self.log.logger.error(f"[{self.name}] Operation failed: {str(e)}")
                should_continue = self.before_retry()
                if not should_continue:
                    raise e # Re-raise the last exception if we are stopping
