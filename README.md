# 🤖 Ollama-RAG 智能客服系统

基于本地 Ollama 模型（Gemma3 + BGE-M3）构建的端到端 RAG 问答系统，支持高并发、低延迟、一键部署。

![架构图](docs/architecture.png) <!-- 可选：用 draw.io 画个简单图 -->

## ✨ 特性
- **本地大模型**：无需 API 费用，完全离线运行
- **高性能检索**：BGE-M3 中文嵌入 + Chroma 向量库
- **智能生成**：Gemma3-4B 生成自然语言回答
- **缓存加速**：Redis 缓存热门问题，QPS 提升 5 倍+
- **工程就绪**：FastAPI + Docker + 压测脚本

## 🚀 快速开始

### 前置条件
- Windows / Linux / macOS
- [Ollama](https://ollama.com/) 已安装并运行
- Python 3.9+
- Redis（可选，但推荐）

### 安装
```bash
git clone https://github.com/yourname/ollama-rag-service.git
cd ollama-rag-service
pip install -r requirements.txt