from loguru import logger
import os
from datetime import datetime

def setup_logger():
    """
    配置日志：按天轮转、保留180天（6个月）、压缩存储
    适配企业审计要求，日志格式包含时间、级别、模块、信息
    """
    # 日志目录（不存在则创建）
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 移除默认日志输出（只保留文件输出）
    logger.remove()
    
    # 添加文件日志（按天轮转）
    logger.add(
        os.path.join(log_dir, "rag-{time:YYYY-MM-DD}.log"),
        rotation="00:00",  # 每天0点创建新日志文件
        retention=180,     # 保留180天（6个月）
        compression="zip", # 旧日志压缩为zip
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module: <10} | {function: <15} | {message}",
        encoding="utf-8",
        enqueue=True,      # 异步日志，提升性能
        backtrace=True,    # 显示完整堆栈信息
        diagnose=True      # 显示变量信息（生产环境可设为False，避免敏感信息泄露）
    )
    
    # 开发环境：同时输出到控制台（生产环境可注释）
    logger.add(
        sink=print,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        encoding="utf-8"
    )
    
    logger.info("📝 日志配置初始化完成（保留180天，按天轮转）")
