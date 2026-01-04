import redis
import json
import hashlib
import httpx
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Optional

# 自动判断是否在 Docker 中（通过环境变量）
IS_DOCKER = os.getenv("IN_DOCKER", "false").lower() == "true"

# Ollama 地址：容器内通过 host.docker.internal 访问宿主机
OLLAMA_HOST = "http://host.docker.internal:11434" if IS_DOCKER else "http://localhost:11434"

# Redis 地址：同上
REDIS_HOST = "host.docker.internal" if IS_DOCKER else "localhost"


class RAGEngine:
    def __init__(self, knowledge_file: str = "data/faq.md"):
        # 初始化 Chroma 向量库
        self.client = chromadb.Client()
        self.collection_name = "faq_rag"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 初始化 Redis（带降级）
        self.redis_client: Optional[redis.Redis] = None
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=False  # 已弃用，可移除
            )
            self.redis_client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print(f"⚠️ Redis not available: {e}. 缓存功能将被禁用。")
            self.redis_client = None  # 明确标记不可用

        # 加载知识库（仅当为空时）
        if self.collection.count() == 0:
            self._load_knowledge(knowledge_file)

    def _get_embedding(self, text: str) -> List[float]:
        """调用 Ollama 的 embed API 获取 bge-m3 嵌入"""
        response = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={
                "model": "bge-m3:latest",
                "input": text
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def _load_knowledge(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = []
        current_q = ""
        current_a = ""
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_q:
                    sections.append((current_q, current_a.strip()))
                current_q = line[3:].strip()
                current_a = ""
            else:
                current_a += line + "\n"
        if current_q:
            sections.append((current_q, current_a.strip()))

        ids = []
        documents = []
        embeddings = []
        for q, a in sections:
            doc = f"问题：{q}\n答案：{a}"
            emb = self._get_embedding(doc)
            doc_id = hashlib.md5(doc.encode()).hexdigest()
            ids.append(doc_id)
            documents.append(doc)
            embeddings.append(emb)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )
        print(f"✅ 已加载 {len(sections)} 条 FAQ 到向量库")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results["documents"][0]

    def ask_with_cache(self, query: str) -> dict:
        cache_key = "rag:" + hashlib.md5(query.encode()).hexdigest()

        # 仅当 Redis 可用时尝试读缓存
        if self.redis_client is not None:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"缓存读取失败: {e}")

        # 执行 RAG
        contexts = self.retrieve(query, top_k=2)
        answer = self.generate_answer(query, contexts)
        result = {"answer": answer}

        # 仅当 Redis 可用时写缓存
        if self.redis_client is not None:
            try:
                self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(result, ensure_ascii=False)
                )
            except Exception as e:
                print(f"缓存写入失败: {e}")

        return result

    def generate_answer(self, query: str, context_list: List[str]) -> str:
        try:
            context = "\n\n".join(context_list)
            prompt = f"""你是一个专业客服助手，请根据以下参考资料回答用户问题。如果资料中没有相关信息，请回答“抱歉，我无法回答该问题”。

参考资料：
{context}

用户问题：{query}

回答："""

            response = httpx.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "gemma3:4b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            else:
                return "抱歉，模型返回格式异常，无法生成答案。"

        except httpx.TimeoutException:
            return "抱歉，模型响应超时，请稍后再试。"
        except httpx.RequestError as e:
            return f"抱歉，连接模型服务失败：{str(e)}"
        except (ValueError, KeyError):
            return "抱歉，模型返回数据格式异常。"
        except Exception as e:
            return f"抱歉，生成答案时发生未知错误：{str(e)}"


if __name__ == '__main__':
    rag = RAGEngine()
    print(rag.generate_answer("测试可以绑定多个手机号吗？", []))