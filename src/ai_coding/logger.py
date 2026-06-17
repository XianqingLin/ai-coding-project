"""日志系统模块.

提供统一的日志配置和日志记录功能.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ai_coding.config import LOG_LEVEL


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
) -> None:
    """配置全局日志系统.
    
    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认使用 config.LOG_LEVEL.
        log_file: 日志文件名, 默认按日期生成.
        log_dir: 日志文件存放目录.
    
    """
    if level is None:
        level = LOG_LEVEL
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 默认日志文件名: logs/ai-coding-YYYY-MM-DD.log
    if log_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = f"ai-coding-{today}.log"
    
    log_file_path = log_path / log_file
    
    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    # 根日志器配置
    root_logger = logging.getLogger("ai_coding")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # 关闭并清除已有处理器，防止文件句柄泄漏
    for h in root_logger.handlers:
        h.close()
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.propagate = False  # 避免重复日志


def get_logger(name: str) -> logging.Logger:
    """获取指定模块的日志记录器.
    
    Args:
        name: 模块名称, 建议使用 __name__.
    
    Returns:
        日志记录器实例.
    
    """
    return logging.getLogger(f"ai_coding.{name}")
