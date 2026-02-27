"""
LLM 接入模块 - 带缓存和限流处理
支持 Kimi API 和 OpenAI API
"""
import os
import json
import requests
import time
from typing import List, Dict, Optional
from functools import lru_cache

class LLMClient:
    """LLM 客户端 - 带限流保护"""
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model: str = "kimi-coding/k2p5", delay: float = 1.0):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or "https://api.moonshot.cn/v1"
        self.model = model
        self.delay = delay  # 请求间隔（秒）
        self.last_request_time = 0
        self.cache = {}  # 简单缓存
        
    def _wait_for_rate_limit(self):
        """等待限流"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            print(f"等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, messages: List[Dict]) -> str:
        """生成缓存 key"""
        return json.dumps(messages, sort_keys=True)
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, 
             use_cache: bool = True) -> str:
        """
        调用 LLM 进行对话（带缓存和限流）
        """
        # 检查缓存
        cache_key = self._get_cache_key(messages)
        if use_cache and cache_key in self.cache:
            print("使用缓存结果")
            return self.cache[cache_key]
        
        # 等待限流
        self._wait_for_rate_limit()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            
            # 存入缓存
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            # 如果有限流，等待更长时间后重试
            if "429" in str(e):
                print("遇到限流，等待 5 秒后重试...")
                time.sleep(5)
                return self.chat(messages, temperature, use_cache=False)
            return ""
    
    def generate(self, prompt: str, system: str = None, use_cache: bool = True) -> str:
        """简化调用方式"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, use_cache=use_cache)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        print("缓存已清空")


# 全局 LLM 客户端（单例）
_llm_client = None

def get_llm_client() -> LLMClient:
    """获取 LLM 客户端（单例模式）"""
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("LLM_API_KEY")
        _llm_client = LLMClient(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            delay=5.0  # 5秒间隔
        )
    return _llm_client


if __name__ == "__main__":
    # 测试
    llm = get_llm_client()
    
    # 第一次调用
    print("第一次调用...")
    r1 = llm.generate("你好，请用一句话介绍知识图谱")
    print(f"回复1: {r1[:50]}...")
    
    # 第二次相同调用（应该走缓存）
    print("\n第二次调用（相同问题）...")
    r2 = llm.generate("你好，请用一句话介绍知识图谱")
    print(f"回复2: {r2[:50]}...")
    
    print(f"\n缓存大小: {len(llm.cache)}")
