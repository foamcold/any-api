import json
import logging
import uuid
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.models.key import OfficialKey
from app.models.user import User
from app.models.system_config import SystemConfig
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

    async def proxy_request(self, body: dict, is_pseudo_stream: bool = False):
        """
        统一处理代理请求的核心方法。
        """
        self.logger.debug(f"[{self.request_id}] 接收到新请求。流程: {self.incoming_format} -> {self.target_provider}")
        
        original_model = body.get("model", "unknown")
        
        client_wants_stream = body.get("stream", False)
        
        if self.is_stream_override is not None:
            is_upstream_stream = self.is_stream_override
        else:
            is_upstream_stream = client_wants_stream
            
        self.logger.debug(f"[{self.request_id}] 原始模型: {original_model}, 客户端期望流式: {client_wants_stream}, 上游将使用流式: {is_upstream_stream}")

        # 只要是伪流请求，就强制进入模拟流
        if is_pseudo_stream and client_wants_stream:
            self.logger.debug(f"[{self.request_id}] 检测到伪流请求，且客户端需要流式响应，启动模拟流响应。")
            return StreamingResponse(
                self._simulate_stream_response(body, original_model),
                media_type="text/event-stream"
            )

        converted_body = body
        converted_body["stream"] = is_upstream_stream
        model_name = original_model

        if self.incoming_format == "openai" and self.target_provider == "gemini":
            self.logger.debug(f"[{self.request_id}] 正在转换请求格式: OpenAI -> Gemini")
            # LLMProxyService 没有预设消息，因此传递 None
            converted_body, model_name = openai_adapter.to_gemini_request(body, preset_messages=None)
            converted_body["stream"] = is_upstream_stream
        elif self.incoming_format == "gemini" and self.target_provider == "openai":
            self.logger.debug(f"[{self.request_id}] 正在转换请求格式: Gemini -> OpenAI")
            converted_body, model_name = gemini_adapter.to_openai_request(body, model_name)
            converted_body["stream"] = is_upstream_stream
        else:
            self.logger.debug(f"[{self.request_id}] 无需转换请求格式 (同构代理)")

        provider = self._get_provider()
        self.logger.debug(f"[{self.request_id}] 正在发送请求到上游服务: {self.target_provider.upper()}")
        upstream_response = None

        if self.target_provider == "openai":
            upstream_response = await provider.create_chat_completion(converted_body)
        elif self.target_provider == "gemini":
            upstream_response = await provider.generate_content(model_name, converted_body)

        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(status_code=upstream_response.status_code, content=json.loads(error_content))

        if is_upstream_stream:
            self.logger.debug(f"[{self.request_id}] 上游响应是流式。开始转换响应流。")
            return StreamingResponse(
                self._stream_response_converter(upstream_response, original_model),
                media_type="text/event-stream"
            )
        else:
            self.logger.debug(f"[{self.request_id}] 上游响应为非流式。正在转换完整响应。")
            response_json = upstream_response.json()
            final_response = self._convert_non_stream_response(response_json, original_model)
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
                else:
                    converted_chunk = chunk_json

                if converted_chunk:
                    yield f"data: {json.dumps(converted_chunk)}\n\n"

            except json.JSONDecodeError:
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
        
        return response_json

    async def proxy_list_models(self, request: Request, system_config: Optional[SystemConfig]):
        """
        代理模型列表请求，并根据系统配置增强列表。
        """
        self.logger.debug(f"[{self.request_id}] 代理模型列表请求。流程: {self.incoming_format} -> {self.target_provider}")
        
        params = request.query_params
        self.logger.debug(f"[{self.request_id}] 传递查询参数: {params}")

        provider = self._get_provider()
        upstream_response = await provider.list_models(params=params)

        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(status_code=upstream_response.status_code, content=json.loads(error_content))

        response_json = upstream_response.json()
        
        if self.target_provider == "gemini" and self.incoming_format == "openai":
            self.logger.debug(f"[{self.request_id}] 模型列表转换: Gemini -> OpenAI")
            final_response = openai_adapter.from_gemini_to_openai_models(response_json)
        elif self.target_provider == "openai" and self.incoming_format == "gemini":
            self.logger.debug(f"[{self.request_id}] 模型列表转换: OpenAI -> Gemini")
            final_response = gemini_adapter.from_openai_to_gemini_models(response_json)
        else:
            final_response = response_json

        if system_config and system_config.pseudo_streaming_enabled:
            self.logger.debug(f"[{self.request_id}] 伪流已启用，正在增强模型列表。")
            pseudo_models = []
            
            original_models = final_response.get("data", final_response.get("models", []))

            for model in original_models:
                pseudo_model = model.copy()
                if "id" in pseudo_model:
                    pseudo_model["id"] = f"伪流/{pseudo_model['id']}"
                if "name" in pseudo_model:
                    model_id = pseudo_model["name"].split('/')[-1]
                    pseudo_model["name"] = f"models/伪流/{model_id}"
                    pseudo_model["displayName"] = f"伪流/{pseudo_model.get('displayName', model_id)}"
                pseudo_models.append(pseudo_model)
            
            if "data" in final_response:
                final_response["data"].extend(pseudo_models)
            elif "models" in final_response:
                final_response["models"].extend(pseudo_models)

        return JSONResponse(content=final_response)

    async def _simulate_stream_response(self, body: dict, original_model: str):
        """
        模拟流式响应：先发送非流式请求，等待时发送空包，收到响应后再发送完整数据。
        """
        is_upstream_stream = False
        converted_body = body
        converted_body["stream"] = is_upstream_stream
        model_name = original_model

        if self.incoming_format == "openai" and self.target_provider == "gemini":
            converted_body, model_name = openai_adapter.to_gemini_request(body)
            converted_body["stream"] = is_upstream_stream
        elif self.incoming_format == "gemini" and self.target_provider == "openai":
            converted_body, model_name = gemini_adapter.to_openai_request(body, model_name)
            converted_body["stream"] = is_upstream_stream
        
        provider = self._get_provider()
        
        upstream_task = None
        if self.target_provider == "openai":
            upstream_task = asyncio.create_task(provider.create_chat_completion(converted_body))
        elif self.target_provider == "gemini":
            upstream_task = asyncio.create_task(provider.generate_content(model_name, converted_body))

        while not upstream_task.done():
            self.logger.debug(f"[{self.request_id}] 伪流: 发送心跳包。")
            
            heartbeat_chunk = {}
            if self.incoming_format == "openai":
                # 构造 OpenAI 格式的空流式块
                heartbeat_chunk = {
                    "id": f"chatcmpl-heartbeat-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": original_model,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]
                }
            elif self.incoming_format == "gemini":
                # 构造 Gemini 格式的空流式块
                heartbeat_chunk = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": ""}],
                            "role": "model"
                        },
                        "finishReason": None,
                        "index": 0
                    }]
                }

            if heartbeat_chunk:
                yield f"data: {json.dumps(heartbeat_chunk)}\n\n"
            
            await asyncio.sleep(1)

        upstream_response = await upstream_task
        
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            error_payload = {"error": json.loads(error_content)}
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        response_json = upstream_response.json()
        final_response = self._convert_non_stream_response(response_json, original_model)
        
        self.logger.debug(f"[{self.request_id}] 伪流: 将完整响应包装为流式块。")

        if self.incoming_format == "openai":
            for i, choice in enumerate(final_response.get("choices", [])):
                chunk = {
                    "id": final_response.get("id"),
                    "object": "chat.completion.chunk",
                    "created": final_response.get("created"),
                    "model": final_response.get("model"),
                    "choices": [{
                        "index": i,
                        "delta": {"content": choice.get("message", {}).get("content", "")},
                        "finish_reason": choice.get("finish_reason")
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
             yield f"data: {json.dumps(final_response)}\n\n"

        yield "data: [DONE]\n\n"