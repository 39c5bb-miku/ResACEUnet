import os
import sys
from loguru import logger
import functools


@functools.lru_cache()
def create_logger(output_dir, dist_rank):
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "<yellow>({file}:{line})</yellow> - <level>{level}</level> - {message}"
    )

    log_file = os.path.join(output_dir, f"log_rank{dist_rank}.txt")
    logger.add(log_file, format=fmt, level="DEBUG", encoding="utf-8")

    if dist_rank == 0:
        logger.add(sys.stdout, format=fmt, level="DEBUG", enqueue=True)

    return logger
