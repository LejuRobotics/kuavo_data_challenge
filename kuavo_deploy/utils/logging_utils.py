import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    """自定义彩色日志格式化器 Customised colour log formatter"""
    
    # 默认样式配置
    DEFAULT_STYLE_CONFIG = {
        'env': {
            'tag': '🤖 ENV',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'green', 'attrs': []},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        },
        'model': {
            'tag': '🧠 MODEL',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'blue', 'attrs': ['bold']},
            'WARNING': {'color': 'magenta', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        },
        'robot': {
            'tag': '🦾 ROBOT',
            'DEBUG': {'color': 'cyan', 'attrs': ['dark']},
            'INFO': {'color': 'green', 'attrs': []},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'red', 'attrs': ['bold', 'underline', 'blink']}
        }
    }
    
    def __init__(self, fmt: str, style_config: Dict = None):
        super().__init__(fmt)
        self.style_config = style_config or self.DEFAULT_STYLE_CONFIG
        self.is_console = False  # 默认为文件输出 File output by default

    def format(self, record):
        # 保存原始消息，因为我们会修改record.msg Cache original message, as it will be modified later on
        original_msg = record.msg
        
        # 获取对应模块的配置 Fetch corresponding module configuration
        source_config = self.style_config.get(record.name, {})
        source_tag = source_config.get('tag', f'📝 {record.name.upper()}')
        style = source_config.get(record.levelname, {'color': 'white', 'attrs': []})
        
        # 构建位置信息 (文件名:行号) - 参考ks_download.py的方法 Construct location info
        location_info = ""
        if hasattr(record, 'pathname') and hasattr(record, 'lineno'):
            fnameline = f"{record.pathname}:{record.lineno}"
            # 截取最后20个字符并右对齐，比ks_download.py稍微长一点以显示更多信息
            # location_info = f" {fnameline[-20:]:>20}"
            location_info = f" {fnameline}"
        
        # 构建消息 Construct messages
        if hasattr(self, 'is_console') and self.is_console:
            # 控制台输出添加颜色 Colour on console output
            colored_message = colored(
                f"{record.levelname}: {original_msg}",
                color=style['color'],
                on_color=style.get('on_color'),
                attrs=style['attrs']
            )
            record.msg = f"{source_tag} | {colored_message} |{location_info} "
        else:
            # 文件输出不添加颜色 No colour for file output
            record.msg = f"{source_tag} | {record.levelname}: {original_msg} | {location_info} "
        # 格式化消息 Formatted message
        formatted_message = super().format(record)
        
        # 恢复原始消息 Restore original message
        record.msg = original_msg
        
        return formatted_message

class LoggerManager:
    def __init__(self, 
                 log_dir: Optional[str] = None, 
                 log_level: str = "INFO",
                 custom_loggers: Optional[Dict] = None,
                 save_to_file: bool = False):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志存储目录
            log_level: 日志级别
            custom_loggers: 自定义logger配置
                例如: {
                    'other': {
                        'tag': '👁️ OHTER',
                        'DEBUG': {'color': 'grey'},
                        'INFO': {'color': 'blue'},
                        ...
                    }
                }
            save_to_file: 是否将日志保存到文件,默认为False
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = self._setup_log_dir(log_dir) if save_to_file else None
        self.loggers = {}
        
        # 合并自定义logger配置
        self.style_config = ColoredFormatter.DEFAULT_STYLE_CONFIG.copy()
        if custom_loggers:
            self.style_config.update(custom_loggers)

        # 如果需要保存到文件,创建统一的文件处理器
        self.file_handler = None
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_handler = logging.FileHandler(
                self.log_dir / f"kuavomimic_{timestamp}.log",
                encoding='utf-8'
            )
            # 文件处理器使用无颜色的formatter
            file_formatter = ColoredFormatter(
                '%(asctime)s - %(message)s',
                style_config=self.style_config
            )
            file_formatter.is_console = False
            self.file_handler.setFormatter(file_formatter)

    def _setup_log_dir(self, log_dir: Optional[str]) -> Path:
        if log_dir is None:
            log_dir = Path.cwd() / 'logs'
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_logger(self, name: str) -> logging.Logger:
        """获取或创建logger"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            logger.handlers.clear()
            
            # 控制台处理器（彩色）
            console_handler = logging.StreamHandler()
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(message)s',
                style_config=self.style_config
            )
            console_formatter.is_console = True  # 标记为控制台输出
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(console_handler)
            if self.file_handler:
                logger.addHandler(self.file_handler)
            
            self.loggers[name] = logger
            
        return self.loggers[name]

# 全局日志管理器实例
_log_manager = None

def get_log_manager(log_dir: Optional[str] = None, 
                   log_level: str = "INFO",
                   custom_loggers: Optional[Dict] = None,
                   save_to_file: bool = False) -> LoggerManager:
    """获取全局日志管理器实例"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LoggerManager(log_dir, log_level, custom_loggers, save_to_file)
    return _log_manager

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, save_to_file: bool = False) -> logging.Logger:
    """
    设置并返回一个命名的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 可选的日志文件路径
        save_to_file: 是否将日志保存到文件,默认为False
        
    Returns:
        配置好的日志记录器
    """
    # 获取全局日志管理器
    log_manager = get_log_manager(log_dir=None, log_level="INFO", custom_loggers=None, save_to_file=save_to_file)
    
    # 获取或创建logger
    logger = log_manager.get_logger(name)
    logger.setLevel(level)
    
    # 如果提供了特定的日志文件，添加额外的文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def highlight_message(logger, message, color="magenta", attrs=None):
    """使用自定义颜色和属性高亮显示消息"""
    if attrs is None:
        attrs = ["bold"]
    print(colored(f">>> {message} <<<", color=color, attrs=attrs))
    return logger.info(message)

def test_logging():
    """测试日志功能"""
    # 创建日志目录
    log_dir = Path.cwd() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"日志文件将保存在: {log_dir}")
    
    # 可选：定义自定义logger
    custom_loggers = {
        'other': {
            'tag': '👋 OTHER',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'cyan', 'attrs': ['bold']},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        }
    }
    
    # 创建日志管理器
    log_manager = LoggerManager(log_dir=str(log_dir), log_level="DEBUG", custom_loggers=custom_loggers)
    
    # 获取loggers
    env_logger = log_manager.get_logger("env")
    model_logger = log_manager.get_logger("model")
    robot_logger = log_manager.get_logger("robot")
    other_logger = log_manager.get_logger("other")  # 自定义logger
    
    # 测试日志
    env_logger.info("环境初始化完成")
    model_logger.warning("模型性能下降")
    robot_logger.info("机器人状态正常")
    other_logger.info("处理相机数据")
    env_logger.error("检测到碰撞风险")
    
    # 测试setup_logger函数 - 不保存到文件
    test_logger = setup_logger("test", logging.DEBUG, save_to_file=False)
    test_logger.debug("这是一条测试日志(仅控制台输出)")
    test_logger.info("测试信息(仅控制台输出)")
    
    # 测试setup_logger函数 - 保存到文件
    test_logger_with_file = setup_logger("test_file", logging.DEBUG, save_to_file=True)
    test_logger_with_file.debug("这是一条测试日志(同时输出到文件)")
    test_logger_with_file.info("测试信息(同时输出到文件)")
    
    # 测试高亮消息
    highlight_message(test_logger, "这是一条高亮消息")
    
    # # 打印日志文件路径
    # log_files = list(log_dir.glob("*.log"))
    # if log_files:
    #     print(f"已创建日志文件: {[str(log_files) for f in log_files]}")
    # else:
    #     print("警告: 未找到日志文件!")

if __name__ == "__main__":
    test_logging()
