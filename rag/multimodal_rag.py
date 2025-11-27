from lazyllm import LazyLLM, MultiModalRAG, CacheModule
from lazyllm.modules import EmbeddingModule, LLMModule, PDFParser, ImageParser, DesensitizeModule
from lazyllm.utils import download_model  # LazyLLM内置模型下载工具（支持镜像）
import os
from loguru import logger

def build_multimodal_rag(redis_url: str = "redis://localhost:6379/0") -> MultiModalRAG:
    """
    构建企业级多模态RAG系统：支持文本+PDF（含表格+图片）+图片+扫描件PDF
    核心特性：多模态解析、跨模态匹配、敏感信息脱敏、缓存优化
    :param redis_url: Redis缓存连接地址
    :return: 多模态RAG实例
    """
    logger.info("📦 初始化企业级多模态RAG系统...")
    
    # 定义模型存储路径（统一管理，避免混乱）
    model_paths = {
        "embedding": "./models/bge-large-zh-v1.5",
        "llm": "./models/deepseek-chat",
        "image": "./models/clip-vit-base-patch32"
    }
    
    # 下载依赖模型（国内网络自动走HF镜像，支持断点续传）
    download_models(model_paths=model_paths)
    
    # 1. 初始化多模态解析模块（企业场景定制化配置）
    ## PDF解析：支持提取文本、表格、图片，适配产品手册/合同场景
    pdf_parser = PDFParser(
        extract_images=True,  # 提取PDF中的图片（如产品截图、架构图）
        layout_analysis=True,  # 启用布局分析（保留标题、表格、正文层级关系）
        table_extraction=True,  # 提取表格（转换为DataFrame，大模型可直接理解）
        ocr_for_scanned=True,  # 对扫描件PDF启用OCR（依赖tesseract+paddleocr）
        ocr_language="ch",  # OCR语言：中文（支持中英混合）
        max_pages=1000  # 支持最大PDF页数（企业手册通常≤500页）
    )
    logger.info("✅ PDF解析模块初始化完成（支持表格+图片+扫描件OCR）")
    
    ## 图片解析：支持视觉特征提取+文字OCR，适配技术图表场景
    image_parser = ImageParser(
        model_name="clip-vit-base-patch32",
        model_path=model_paths["image"],
        device="cuda:0" if LazyLLM.is_cuda_available() else "cpu",
        quantize="8bit",  # 8bit量化：显存占用从1.7GB降至0.9GB
        ocr=True,  # 启用OCR识别图片中的文字（技术图表含大量文字）
        ocr_model="paddleocr",  # OCR引擎：paddleocr（中文准确率比tesseract高8%）
        resize_max_size=1024  # 图片最大尺寸：避免高清图片占用过多显存
    )
    logger.info("✅ 图片解析模块初始化完成（支持视觉特征+OCR文字识别）")
    
    ## 敏感信息脱敏模块（企业场景必需，避免泄露机密）
    desensitize = DesensitizeModule(
        types=["phone", "address", "id_card", "company_seal"],  # 脱敏类型：电话、地址、身份证、企业公章
        replace_with="[***]",  # 替换符：统一用[***]隐藏敏感信息
        strict_mode=True  # 严格模式：宁可误脱敏，不可漏脱敏
    )
    logger.info("✅ 敏感信息脱敏模块初始化完成（支持4类敏感信息屏蔽）")
    
    # 2. 初始化嵌入模型和大模型（复用性能优化配置）
    from rag.lazyllm_optimized import build_optimized_rag
    base_rag = build_optimized_rag(redis_url=redis_url)
    embedding = base_rag.embedding
    llm = base_rag.llm
    logger.info("✅ 复用性能优化后的嵌入模型和大模型")
    
    # 3. 构建多模态RAG系统（整合多模态组件，支持跨模态匹配）
    multimodal_rag = MultiModalRAG(
        embedding=embedding,
        llm=llm,
        vector_db="chroma",
        db_path="./multimodal_vector_db",  # 多模态向量库独立存储，避免和纯文本冲突
        parsers=[pdf_parser, image_parser, desensitize],  # 执行顺序：解析→脱敏（先解析再脱敏）
        cross_modal_matching=True,  # 支持跨模态匹配（文本查图片、图片查文本）
        cache_module=base_rag.modules[0],  # 复用Redis缓存模块
        prompt_template="""基于以下多模态参考数据（文本+表格+图片），回答用户问题：
{context}
用户问题：{query}
回答要求：
1. 文本内容简洁分点；
2. 表格数据按"表头：值"格式说明；
3. 图片内容描述核心信息（如"架构图中核心组件为MySQL+Redis"）；
4. 标注数据来源（文件路径/页码/图片名称）；
5. 无相关信息时说明"暂无相关多模态数据"。"""
    )
    logger.info("✅ 企业级多模态RAG初始化完成（支持文本+PDF+图片+扫描件）")
    return multimodal_rag

def download_models(model_paths: dict):
    """
    下载多模态所需模型（支持国内HF镜像、断点续传）
    :param model_paths: 模型名称→存储路径的映射
    """
    logger.info("📥 开始下载多模态所需模型（国内自动走HF镜像）")
    
    # 下载嵌入模型（BGE中文）
    if not os.path.exists(model_paths["embedding"]):
        logger.info(f"正在下载嵌入模型：BAAI/bge-large-zh-v1.5")
        download_model(
            repo_id="BAAI/bge-large-zh-v1.5",
            local_dir=model_paths["embedding"],
            mirror="hf-mirror.com",  # 国内镜像，避免超时
            resume=True  # 断点续传：下载中断后可继续
        )
        logger.info("✅ 嵌入模型下载完成")
    else:
        logger.info(f"嵌入模型已存在：{model_paths['embedding']}（跳过下载）")
    
    # 下载大模型（DeepSeek-Chat）
    if not os.path.exists(model_paths["llm"]):
        logger.info(f"正在下载大模型：deepseek-ai/deepseek-chat")
        download_model(
            repo_id="deepseek-ai/deepseek-chat",
            local_dir=model_paths["llm"],
            mirror="hf-mirror.com",
            resume=True,
            trust_remote_code=True  # 需信任远程代码（DeepSeek模型要求）
        )
        logger.info("✅ 大模型下载完成")
    else:
        logger.info(f"大模型已存在：{model_paths['llm']}（跳过下载）")
    
    # 下载图片模型（CLIP）
    if not os.path.exists(model_paths["image"]):
        logger.info(f"正在下载图片模型：openai/clip-vit-base-patch32")
        download_model(
            repo_id="openai/clip-vit-base-patch32",
            local_dir=model_paths["image"],
            mirror="hf-mirror.com",
            resume=True
        )
        logger.info("✅ 图片模型下载完成")
    else:
        logger.info(f"图片模型已存在：{model_paths['image']}（跳过下载）")

# 测试代码（本地验证多模态功能）
if __name__ == "__main__":
    # 初始化多模态RAG
    multimodal_rag = build_multimodal_rag()
    
    # 加载示例数据（确保data目录下有对应文件，或替换为实际数据路径）
    data_paths = [
        "./data/企业知识库文本.txt",
        "./data/2024产品功能手册.pdf",
        "./data/系统架构流程图.png",
        "./data/合作合同扫描件.pdf"
    ]
    logger.info(f"📥 开始加载多模态数据（共{len(data_paths)}个文件）")
    try:
        multimodal_rag.load_data(data_paths, batch_size=10)
        logger.info("✅ 多模态数据加载完成")
    except Exception as e:
        logger.error(f"❌ 数据加载失败：{str(e)}，请检查文件路径是否正确")
        exit(1)
    
    # 测试1：文本查询PDF表格数据
    query1 = "2024产品的API调用频率限制是多少？"
    logger.info(f"\n🚀 测试1：文本查询PDF表格 → {query1}")
    result1 = multimodal_rag(query1)
    logger.info(f"✅ 结果1：\n{result1}")
    
    # 测试2：跨模态查询（文本查图片内容）
    query2 = "系统架构流程图中，核心数据存储组件是什么？"
    logger.info(f"\n🚀 测试2：跨模态查询 → {query2}")
    result2 = multimodal_rag(query2)
    logger.info(f"✅ 结果2：\n{result2}")
    
    # 测试3：查询扫描件PDF中的内容
    query3 = "合作合同扫描件中，服务期限是多久？"
    logger.info(f"\n🚀 测试3：扫描件查询 → {query3}")
    result3 = multimodal_rag(query3)
    logger.info(f"✅ 结果3：\n{result3}")
    
    # 测试4：敏感信息脱敏效果
    query4 = "合同中的联系人电话是多少？"
    logger.info(f"\n🚀 测试4：敏感信息脱敏 → {query4}")
    result4 = multimodal_rag(query4)
    logger.info(f"✅ 结果4：\n{result4}")
