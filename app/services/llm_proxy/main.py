import json
import logging
import uuid
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.models.key import OfficialKey
from app.models.user import User
from app.services.llm_proxy.adapters import openai_adapter, gemini_adapter
from app.services.llm_proxy.providers import openai_provider, gemini_provider
from app.core.config import settings

class LLMProxyService:
    def __init__(
        self,
        db: AsyncSession,
        official_key_obj: OfficialKey,
        user: Optional[User],
        incoming_format: str,
        target_provider: str,
        is_stream_override: Optional[bool] = None
    ):
        self.db = db
        self.key_obj = official_key_obj
        self.user = user
        self.incoming_format = incoming_format
        self.target_provider = target_provider
        self.is_stream_override = is_stream_override
        self.request_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)

    async def proxy_request(self, request: Request):
        """
        统一处理代理请求的核心方法。
        """
        self.logger.debug(f"[{self.request_id}] 接收到新请求。流程: {self.incoming_format} -> {self.target_provider}")
        
        body = await request.json()
        original_model = body.get("model", "unknown")
        
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            is_stream = body.get("stream", False)
            
        self.logger.debug(f"[{self.request_id}] 原始模型: {original_model}, 是否流式: {is_stream}")

        # 1. 请求转换 (Incoming -> Target)
        # 将 is_stream 标志传递给转换后的 body，以便 provider 可以使用它
        converted_body = body
        converted_body["stream"] = is_stream
        model_name = original_model

        if self.incoming_format == "openai" and self.target_provider == "gemini":
            self.logger.debug(f"[{self.request_id}] 正在转换请求格式: OpenAI -> Gemini")
            converted_body, model_name = openai_adapter.to_gemini_request(body)
            converted_body["stream"] = is_stream
        elif self.incoming_format == "gemini" and self.target_provider == "openai":
            self.logger.debug(f"[{self.request_id}] 正在转换请求格式: Gemini -> OpenAI")
            converted_body, model_name = gemini_adapter.to_openai_request(body, model_name)
            converted_body["stream"] = is_stream
        else:
            self.logger.debug(f"[{self.request_id}] 无需转换请求格式 (同构代理)")


        # 2. 选择上游Provider并发送请求
        provider = self._get_provider()
        self.logger.debug(f"[{self.request_id}] 正在发送请求到上游服务: {self.target_provider.upper()}")
        upstream_response = None

        if self.target_provider == "openai":
            upstream_response = await provider.create_chat_completion(converted_body)
        elif self.target_provider == "gemini":
            upstream_response = await provider.generate_content(model_name, converted_body)

        # 3. 处理上游响应
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(status_code=upstream_response.status_code, content=json.loads(error_content))

        # 4. 响应转换 (Target -> Incoming)
        # 探测响应是否为流式
        content_type = upstream_response.headers.get("content-type", "")
        is_upstream_stream = "text/event-stream" in content_type
        
        if is_upstream_stream:
            self.logger.debug(f"[{self.request_id}] 上游响应是流式。开始转换响应流。")
            return StreamingResponse(
                self._stream_response_converter(upstream_response, original_model), # Use original_model for response
                media_type="text/event-stream"
            )
        else:
            self.logger.debug(f"[{self.request_id}] 上游响应为非流式。正在转换完整响应。")
            response_json = upstream_response.json() # .json() is a sync method after the response is read
            final_response = self._convert_non_stream_response(response_json, original_model) # Use original_model for response
            self.logger.debug(f"[{self.request_id}] 正在发送最终转换后的响应给客户端。")
            return JSONResponse(content=final_response)

    def _get_provider(self):
        """根据目标提供商选择并实例化对应的Provider。"""
        if self.target_provider == "openai":
            return openai_provider.OpenAIProvider(api_key=self.key_obj.key, base_url=settings.OPENAI_BASE_URL)
        elif self.target_provider == "gemini":
            return gemini_provider.GeminiProvider(api_key=self.key_obj.key, base_url=settings.GEMINI_BASE_URL)
        raise ValueError(f"Unsupported target provider: {self.target_provider}")

    async def _stream_response_converter(self, upstream_response, original_model: str):
        """将上游的流式响应转换为客户端所需的格式。"""
        async for chunk in upstream_response.aiter_bytes():
            if not chunk.strip():
                continue
            
            # 移除 "data: " 前缀
            if chunk.startswith(b'data: '):
                chunk_content = chunk[6:]
            else:
                chunk_content = chunk
            
            try:
                chunk_json = json.loads(chunk_content)
                converted_chunk = None
                
                if self.target_provider == "gemini" and self.incoming_format == "openai":
                    self.logger.debug(f"[{self.request_id}] 流式块转换: Gemini -> OpenAI")
                    converted_chunk = openai_adapter.from_gemini_stream_chunk(chunk_json, original_model)
                elif self.target_provider == "openai" and self.incoming_format == "gemini":
                    self.logger.debug(f"[{self.request_id}] 流式块转换: OpenAI -> Gemini")
                    converted_chunk = gemini_adapter.from_openai_stream_chunk(chunk_json)
                else: # 同构透传
                    converted_chunk = chunk_json

                if converted_chunk:
                    yield f"data: {json.dumps(converted_chunk)}\n\n"

            except json.JSONDecodeError:
                # 忽略无法解析的块
                continue
        
        yield "data: [DONE]\n\n"

    def _convert_non_stream_response(self, response_json: dict, original_model: str) -> dict:
        """将上游的非流式响应转换为客户端所需的格式。"""
        if self.target_provider == "gemini" and self.incoming_format == "openai":
            self.logger.debug(f"[{self.request_id}] 响应转换: Gemini -> OpenAI")
            return openai_adapter.from_gemini_response(response_json, original_model, is_stream=False)
        elif self.target_provider == "openai" and self.incoming_format == "gemini":
            self.logger.debug(f"[{self.request_id}] 响应转换: OpenAI -> Gemini")
            return gemini_adapter.from_openai_response(response_json)
        
        # 同构透传
        return response_json