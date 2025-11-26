# lazyllm-rag-industrial
LazyLLM工业级RAG系统实战代码：包含性能优化、多模态检索、Docker/K8s高可用部署等生产级方案

## 项目介绍
基于 LazyLLM 构建的企业级 RAG 系统，解决「响应慢、多模态不支持、稳定性差」三大核心问题，实测：
- 响应速度提升 75%（12.5秒 → 3.1秒）
- 显存占用降低 40%（量化优化）
- 支持文本+PDF+图片+扫描件混合检索
- 部署可用性达 99.9%（Docker/K8s 高可用方案）

## 适用场景
企业知识库、产品咨询、技术支持、合同检索等通用场景，支持日均 5 万次查询（RTX 3080 环境）。

## 快速开始（Docker Compose 单机部署）
### 1. 环境要求
- Docker 20.10+（支持 GPU，需安装 NVIDIA Docker Runtime）
- GPU 显存 ≥ 8GB（推荐 RTX 3080/4090）
- 磁盘空间 ≥ 100GB（模型+数据+向量库）

### 2. 部署步骤
```bash
# 1. 克隆仓库
git clone https://github.com/osarch/lazyllm-rag-industrial.git
cd lazyllm-rag-industrial

# 2. 下载模型（国内自动走HF镜像，避免超时）
python scripts/download_models.py

# 3. 启动服务（后台运行）
cd docker
docker-compose up -d

# 4. 验证服务
# - RAG接口文档：http://localhost:8000/docs
# - Grafana监控面板：http://localhost:3000（账号：admin，密码：admin123）
