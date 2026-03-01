"""
LLM 接入模块 - 使用 OpenAI 客户端
支持通义千问、Kimi 等
"""
import os
import time
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("安装 openai 库...")
    import subprocess
    subprocess.run(["pip", "install", "openai", "-q"])
    from openai import OpenAI

class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model: str = None, delay: float = 1.0):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.delay = delay
        self.last_request_time = 0
        self.cache = {}
        
        # 默认使用通义千问
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = model or "qwen-plus"
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"使用模型: {self.model}")
        print(f"Base URL: {self.base_url}")
    
    def _wait_for_rate_limit(self):
        """等待限流"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            print(f"等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, 
             use_cache: bool = True) -> str:
        """调用 LLM"""
        cache_key = str(messages)
        if use_cache and cache_key in self.cache:
            print("使用缓存结果")
            return self.cache[cache_key]
        
        self._wait_for_rate_limit()
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            result = completion.choices[0].message.content
            
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return f"调用失败: {e}"
    
    def generate(self, prompt: str, system: str = None, use_cache: bool = True) -> str:
        """简化调用"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, use_cache=use_cache)


# 全局客户端
_llm_client = None

def get_llm_client(api_key: str = None) -> LLMClient:
    """获取 LLM 客户端"""
    global _llm_client
    if _llm_client is None:
        key = api_key or os.getenv("LLM_API_KEY")
        _llm_client = LLMClient(api_key=key, delay=1.0)
    return _llm_client


if __name__ == "__main__":
    # 测试
    llm = get_llm_client()
    response = llm.generate("你好，请用一句话介绍知识图谱")
    print(f"回复: {response}")
