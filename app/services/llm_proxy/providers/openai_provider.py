"""
负责与OpenAI上游服务进行通信。
"""
import httpx
from typing import Dict, Any

class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    def get_headers(self) -> Dict[str, str]:
        """
        构建访问OpenAI API所需的请求头。
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def create_chat_completion(self, body: Dict[str, Any]):
        """
        向上游发送 chat completion 请求。
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = self.get_headers()
        is_stream = body.get("stream", False)
        
        request = self.client.build_request("POST", url, headers=headers, json=body)
        
        try:
            response = await self.client.send(request, stream=is_stream)
            response.raise_for_status()  # 如果状态码是4xx或5xx，则抛出异常
            return response
        except httpx.HTTPStatusError as e:
            # 将上游的错误响应原样返回，以便上层可以进行处理
            await e.response.aread() # 确保内容已异步读取
            return e.response
        except httpx.RequestError as e:
            # 对于网络层面的错误，构建一个httpx.Response对象以保持接口统一
            error_body = {"error": {"message": f"Upstream request failed: {e}", "type": "connection_error"}}
            return httpx.Response(status_code=502, json=error_body, request=request)
