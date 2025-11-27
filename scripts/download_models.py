"""
模型自动下载脚本：支持国内HF镜像、断点续传，避免用户手动下载
运行命令：python scripts/download_models.py
"""
from lazyllm.utils import download_model
import os
from loguru import logger

def main():
    # 定义模型列表和存储路径
    models = [
        {
            "repo_id": "BAAI/bge-large-zh-v1.5",
            "local_dir": "./models/bge-large-zh-v1.5",
            "description": "中文嵌入模型（多模态RAG核心）"
        },
        {
            "repo_id": "deepseek-ai/deepseek-chat",
            "local_dir": "./models/deepseek-chat",
            "description": "中文大模型（回答生成）",
            "trust_remote_code": True
        },
        {
            "repo_id": "openai/clip-vit-base-patch32",
            "local_dir": "./models/clip-vit-base-patch32",
            "description": "图片特征提取模型（多模态检索）"
        }
    ]
    
    logger.info("🚀 开始下载LazyLLM RAG所需模型（国内自动走HF镜像）")
    logger.info(f"共需下载{len(models)}个模型，总大小约30GB，请确保磁盘空间充足")
    
    for model in models:
        repo_id = model["repo_id"]
        local_dir = model["local_dir"]
        description = model["description"]
        
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
            logger.info(f"✅ {description}已存在（{local_dir}），跳过下载")
            continue
        
        logger.info(f"\n📥 正在下载：{repo_id}（{description}）")
        try:
            download_model(
                repo_id=repo_id,
                local_dir=local_dir,
                mirror="hf-mirror.com",  # 国内镜像，解决下载超时
                resume=True,  # 断点续传
                trust_remote_code=model.get("trust_remote_code", False)
            )
            logger.info(f"✅ {description}下载完成（存储路径：{local_dir}）")
        except Exception as e:
            logger.error(f"❌ {description}下载失败：{str(e)}", exc_info=True)
            logger.warning("⚠️  建议：1. 检查网络连接；2. 手动下载模型后放到对应目录；3. 重新运行脚本（支持断点续传）")
    
    logger.info("\n🎉 所有模型下载完成！可开始部署RAG服务")
    logger.info("部署命令参考：")
    logger.info("1. Docker Compose单机：cd docker && docker-compose up -d")
    logger.info("2. K8s集群：cd k8s && kubectl apply -f rag-deployment.yaml")

if __name__ == "__main__":
    # 初始化日志（控制台输出）
    logger.remove()
    logger.add(
        sink=print,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        encoding="utf-8"
    )
    main()
