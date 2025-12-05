"""
负责与Google Gemini上游服务进行通信。
"""
import httpx
from typing import Dict, Any

class GeminiProvider:
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    def get_headers(self) -> Dict[str, str]:
        """
        构建访问Gemini API所需的请求头。
        """
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    async def generate_content(self, model: str, body: Dict[str, Any]):
        """
        向上游发送 generateContent (或 streamGenerateContent) 请求。
        """
        # 检查 body 中是否有 stream 字段，以确定是否是客户端请求的流式响应
        # 注意：转换后的 body 可能不直接包含 stream 字段，我们需要从原始请求中判断
        is_stream = body.pop("stream", False)
        action = "streamGenerateContent?alt=sse" if is_stream else "generateContent"
        
        # 确保模型名称不包含 "models/" 前缀
        model_id = model.split('/')[-1]

        url = f"{self.base_url}/v1beta/models/{model_id}:{action}"
        headers = self.get_headers()
        
        # Gemini API 的 body 不包含 model 字段
        downstream_body = body.copy()
        if "model" in downstream_body:
            del downstream_body["model"]
            
        request = self.client.build_request("POST", url, headers=headers, json=downstream_body)
        
        try:
            response = await self.client.send(request, stream=is_stream)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            await e.response.aread()
            return e.response
        except httpx.RequestError as e:
            error_body = {"error": {"message": f"Upstream request failed: {e}", "type": "connection_error"}}
            return httpx.Response(status_code=502, json=error_body, request=request)
