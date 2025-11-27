from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import os
from contextlib import asynccontextmanager

# 导入路由
from api.router.rag_router import rag_router
# 导入日志配置
from api.logging import setup_logger

# 初始化日志（项目启动时执行）
setup_logger()

# 全局RAG实例（避免重复初始化，提升性能）
global_rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：启动时初始化RAG，关闭时释放资源
    """
    global global_rag
    logger.info("🚀 启动LazyLLM RAG服务...")
    try:
        # 初始化多模态RAG（生产环境用多模态版本，兼容文本/PDF/图片）
        from rag.multimodal_rag import build_multimodal_rag
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        global_rag = build_multimodal_rag(redis_url=redis_url)
        logger.info("✅ RAG实例初始化成功（支持多模态检索）")
        
        # 加载示例数据（可选，生产环境可注释，手动加载实际数据）
        data_dir = "./data"
        if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
            logger.info(f"📥 加载示例数据（目录：{data_dir}）")
            global_rag.load_data(data_dir, batch_size=10)
            logger.info("✅ 示例数据加载完成")
    except Exception as e:
        logger.error(f"❌ RAG实例初始化失败：{str(e)}", exc_info=True)
        raise e
    yield
    # 关闭时释放资源
    logger.info("🔌 关闭RAG服务，释放资源...")
    global_rag = None

# 创建FastAPI应用
app = FastAPI(
    title="LazyLLM 工业级RAG服务",
    description="支持文本+PDF+图片多模态检索，性能优化+高可用部署",
    version="1.0.0",
    lifespan=lifespan
)

# 注册路由（RAG查询接口）
app.include_router(rag_router, prefix="/v1/rag", tags=["RAG检索接口"])

# 健康检查接口（Docker/K8s探针使用）
@app.get("/health", summary="健康检查")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "healthy", "message": "RAG服务正常运行"})

# 就绪检查接口（K8s就绪探针）
@app.get("/ready", summary="就绪检查")
async def ready_check():
    if global_rag is None:
        raise HTTPException(status_code=503, detail="RAG实例未初始化完成")
    return JSONResponse(status_code=200, content={"status": "ready", "message": "RAG服务可接收请求"})

# 根路径接口
@app.get("/", summary="根路径")
async def root():
    return JSONResponse(
        content={
            "message": "欢迎使用LazyLLM工业级RAG服务",
            "docs_url": "/docs",
            "version": "1.0.0"
        }
    )

if __name__ == "__main__":
    # 本地开发启动（生产环境用Docker/K8s启动）
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("WORKERS", 4)),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
