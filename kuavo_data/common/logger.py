from datetime import datetime

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        # Clear log file on initialization
        with open(self.log_file, "w") as f:
            f.write("")

    def log(self, text):
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")