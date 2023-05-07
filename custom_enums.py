from enum import Enum, auto


class FileOption(Enum):
    TRAINING_ACCURACY = auto(),
    TRAINING_LOSS = auto(),
    VALIDATION_ACCURACY = auto(),
    VALIDATION_LOSS = auto(),

    def get_file_option_name(self):
        return self.name.lower()
