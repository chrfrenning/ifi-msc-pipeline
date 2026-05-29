import logging

#
# Standard'ish python logging
#

def create_logger(name):
    logger = logging.getLogger(name)
    setup_logging(logger)
    return logger

def setup_logging(logger):
    logger.setLevel(logging.DEBUG)

    # Handler for INFO and above to stdout
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno >= logging.INFO)
    info_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    info_handler.stream = logging.sys.stderr

    # Handler for below INFO (DEBUG) to stderr
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno < logging.INFO)
    debug_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    debug_handler.stream = logging.sys.stderr

    logger.handlers = []  # Remove any default handlers
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)


#
# Custom stdout printer with indentation
#

indent = 0

def printer(str):
    global indent
    print(f"{'  ' * indent}{str}")

class IndentBlock:
    def __enter__(self):
        global indent
        indent += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        global indent
        indent -= 1

def printer_indent():
    return IndentBlock()