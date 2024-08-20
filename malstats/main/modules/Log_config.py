import logging
from .filenames import logging_path

# Define the base configuration
logging.basicConfig(level=logging.WARNING,
                    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


def setup_loggers():
    base_logger = logging.getLogger("Nyanpasutats")
    base_logger.setLevel(logging.INFO)
    base_formatter = logging.Formatter("[%(asctime)s] %(name)s:%(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logging_path.mkdir(exist_ok=True)

    # view logger
    view_logger = logging.getLogger("Nyanpasutats.view")
    view_logger.setLevel(logging.INFO)
    view_file_handler = logging.FileHandler(logging_path / "views.log", mode='a')
    view_file_handler.setFormatter(base_formatter)
    view_logger.addHandler(view_file_handler)


print("Setting up loggers")
setup_loggers()
