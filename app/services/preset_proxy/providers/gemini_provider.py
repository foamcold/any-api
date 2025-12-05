import httpx
import logging
import json
from typing import Dict, Any

class GeminiProvider:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    async def generate_content(self, model: str, body: Dict[str, Any], is_stream: bool) -> httpx.Response:
        client = httpx.AsyncClient()
        action = "streamGenerateContent?alt=sse" if is_stream else "generateContent"
        
        url = f"{self.base_url}/v1beta/models/{model}:{action}"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        logging.debug(f"发送到上游 (Gemini) 的请求: URL={url}, Body={json.dumps(body, ensure_ascii=False)}")

        request = client.build_request("POST", url, headers=headers, json=body, timeout=300)
        
        try:
            response = await client.send(request, stream=is_stream)
            return response
        except httpx.RequestError as e:
            error_body = {"error": {"message": f"Upstream request failed: {e}", "type": "connection_error"}}
            return httpx.Response(status_code=502, json=error_body, request=request)

    async def list_models(self, params=None) -> httpx.Response:
        client = httpx.AsyncClient()
        url = f"{self.base_url}/v1beta/models"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        request = client.build_request("GET", url, headers=headers, params=params, timeout=300)
        
        try:
            response = await client.send(request)
            return response
        except httpx.RequestError as e:
            error_body = {"error": {"message": f"Upstream request failed: {e}", "type": "connection_error"}}
            return httpx.Response(status_code=502, json=error_body, request=request)