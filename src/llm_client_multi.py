"""
LLM 接入模块 - 支持多平台
Kimi、通义千问等
"""
import os
import json
import requests
import time
from typing import List, Dict, Optional

class LLMClient:
    """LLM 客户端 - 支持多平台"""
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model: str = None, delay: float = 1.0, platform: str = "auto"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.delay = delay
        self.last_request_time = 0
        self.cache = {}
        
        # 自动检测平台
        if platform == "auto":
            platform = self._detect_platform(self.api_key)
        self.platform = platform
        
        # 设置平台参数
        if platform == "qianwen":
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model = model or "qwen-plus"
        elif platform == "kimi":
            self.base_url = base_url or "https://api.moonshot.cn/v1"
            self.model = model or "kimi-coding/k2p5"
        else:
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-3.5-turbo"
        
        print(f"使用平台: {platform}, 模型: {self.model}")
    
    def _detect_platform(self, api_key: str) -> str:
        """根据 API Key 检测平台"""
        if api_key.startswith("sk-") and len(api_key) > 50:
            # 通义千问 Key 通常较长
            return "qianwen"
        elif "kimi" in api_key.lower():
            return "kimi"
        else:
            return "openai"
    
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
    
    def _call_qianwen(self, messages: List[Dict], temperature: float) -> str:
        """调用通义千问 API (兼容模式)"""
        # 兼容模式使用 api-key 认证
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "api-key": self.api_key,
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
        return data["choices"][0]["message"]["content"]
    
    def _call_kimi(self, messages: List[Dict], temperature: float) -> str:
        """调用 Kimi API"""
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
        return data["choices"][0]["message"]["content"]
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, 
             use_cache: bool = True) -> str:
        """调用 LLM"""
        cache_key = self._get_cache_key(messages)
        if use_cache and cache_key in self.cache:
            print("使用缓存结果")
            return self.cache[cache_key]
        
        self._wait_for_rate_limit()
        
        try:
            if self.platform == "qianwen":
                result = self._call_qianwen(messages, temperature)
            elif self.platform == "kimi":
                result = self._call_kimi(messages, temperature)
            else:
                result = self._call_kimi(messages, temperature)  # 默认用 OpenAI 格式
            
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("遇到限流，等待 5 秒后重试...")
                time.sleep(5)
                return self.chat(messages, temperature, use_cache=False)
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
        _llm_client = LLMClient(api_key=key, delay=2.0)
    return _llm_client
