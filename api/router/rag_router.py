from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
from typing import Optional

# 导入全局RAG实例
from api.server import global_rag

# 创建路由实例
router = APIRouter()

# 初始化限流（企业场景防恶意请求）
limiter = Limiter(key_func=get_remote_address)
router.state.limiter = limiter
router.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 请求模型（校验输入）
class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Query(default=8, ge=1, le=20, description="检索返回文档数")
    similarity_threshold: Optional[float] = Query(default=0.6, ge=0.1, le=0.9, description="相似度阈值")

# 响应模型（规范输出）
class RAGQueryResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: dict = {
        "query": "",
        "answer": "",
        "data_source": [],  # 数据来源（文件路径）
        "response_time": 0.0,  # 响应时间（秒）
        "cache_hit": False  # 是否命中缓存
    }

@router.post("/query", summary="RAG检索查询", response_model=RAGQueryResponse)
@limiter.limit("20/minute")  # 限制单IP 20次/分钟（可根据需求调整）
async def rag_query(
    request: RAGQueryRequest,
    client_ip: str = Depends(get_remote_address)
):
    """
    多模态RAG检索接口：支持文本查询PDF/图片/文本中的内容
    - query: 查询文本（必填）
    - top_k: 检索返回文档数（1-20）
    - similarity_threshold: 相似度阈值（0.1-0.9）
    """
    import time
    start_time = time.time()
    try:
        logger.info(f"📩 接收RAG查询：IP={client_ip}, query={request.query[:50]}..., top_k={request.top_k}")
        
        # 校验RAG实例是否就绪
        if global_rag is None:
            logger.error(f"❌ RAG实例未初始化，查询失败：{request.query[:50]}...")
            raise HTTPException(status_code=503, detail="RAG服务未就绪，请稍后重试")
        
        # 执行查询（覆盖默认参数）
        result = global_rag(
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # 计算响应时间
        response_time = round(time.time() - start_time, 2)
        
        # 判断是否命中缓存（从CacheModule中获取）
        cache_hit = False
        for module in global_rag.modules:
            if hasattr(module, "hit_rate") and hasattr(module, "_last_hit"):
                cache_hit = module._last_hit
                break
        
        # 构造响应（提取数据来源，简化输出）
        data_source = []
        if hasattr(result, "sources"):
            data_source = [source["path"] for source in result.sources[:3]]  # 最多返回3个来源
        
        logger.info(f"✅ RAG查询成功：IP={client_ip}, query={request.query[:50]}..., 耗时={response_time}秒, 缓存命中={cache_hit}")
        
        return RAGQueryResponse(
            data={
                "query": request.query,
                "answer": result.strip() if isinstance(result, str) else str(result),
                "data_source": data_source,
                "response_time": response_time,
                "cache_hit": cache_hit
            }
        )
    except Exception as e:
        response_time = round(time.time() - start_time, 2)
        logger.error(f"❌ RAG查询失败：IP={client_ip}, query={request.query[:50]}..., 耗时={response_time}秒, 错误={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"查询失败：{str(e)}"
        ) from e

@router.get("/stats", summary="获取服务统计信息")
async def get_rag_stats():
    """
    获取RAG服务运行统计：缓存命中率、模型信息等
    """
    if global_rag is None:
        raise HTTPException(status_code=503, detail="RAG服务未就绪")
    
    # 提取缓存命中率
    cache_hit_rate = 0.0
    for module in global_rag.modules:
        if hasattr(module, "hit_rate"):
            cache_hit_rate = round(module.hit_rate, 4)
            break
    
    # 提取模型信息
    llm_model = global_rag.llm.model_name if hasattr(global_rag, "llm") else "unknown"
    embedding_model = global_rag.embedding.model_name if hasattr(global_rag, "embedding") else "unknown"
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "vector_db": global_rag.vector_db if hasattr(global_rag, "vector_db") else "unknown",
            "supported_data_types": ["text", "pdf", "image", "scanned_pdf"]
        }
    }
