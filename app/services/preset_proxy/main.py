import logging
import uuid
import json
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.models.key import ExclusiveKey, OfficialKey
from app.models.user import User
from app.models.channel import Channel
from app.models.preset import Preset
from app.services.preset_proxy.adapters import openai_adapter
from app.services.preset_proxy.providers import openai_provider, gemini_provider
from app.services.variable_service import variable_service
from app.services.gemini_service import gemini_service
from app.services.regex_service import regex_service
from app.services import universal_adapter

class PresetProxyService:
    def __init__(
        self,
        db: AsyncSession,
        exclusive_key_obj: ExclusiveKey,
        user: User,
        incoming_format: str,
        is_stream_override: Optional[bool] = None,
    ):
        self.db = db
        self.exclusive_key_obj = exclusive_key_obj
        self.user = user
        self.incoming_format = incoming_format
        self.is_stream_override = is_stream_override
        self.request_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"[{self.request_id}] PresetProxyService initialized for incoming format: {self.incoming_format}")

    async def proxy_request(self, request: Request):
        self.logger.debug(f"[{self.request_id}] 接收到新的预设模式请求。")

        # 1. 获取渠道和官方密钥
        channel = self.exclusive_key_obj.channel
        if not channel:
            raise Exception("专属密钥未绑定渠道")
        
        official_key = await gemini_service.get_active_key(self.db, channel.id)

        # 2. 获取预设并处理变量
        preset = self.exclusive_key_obj.preset
        preset_messages = []
        if preset:
            messages = json.loads(preset.content)
            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = variable_service.parse_variables(msg["content"])
            preset_messages = messages

        # 3. 解析请求体并转换为通用格式
        body = await request.json()
        
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            is_stream = body.get("stream", False)

        general_request = universal_adapter.to_general_openai_request(body, self.incoming_format, is_stream)

        # 4. 应用前置正则
        if preset:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset.id, PresetRegexRule.type == "pre")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()
            
            for msg in general_request.messages:
                if isinstance(msg.content, str):
                    msg.content = regex_service.process(msg.content, rules)

        # 5. 注入预设
        general_request.messages = [msg for msg in preset_messages] + [msg.model_dump() for msg in general_request.messages]

        # 6. 转换为目标渠道格式
        target_format = channel.type
        self.logger.debug(f"[{self.request_id}] Target channel format: {target_format}")
        
        if target_format == "gemini":
            converted_body, model_name = openai_adapter.to_gemini_request({"messages": general_request.messages, **general_request.model_dump(exclude={"messages"})}, [])
        else: # OpenAI or compatible
            converted_body = general_request.model_dump(exclude_none=True)
            model_name = general_request.model

        # 7. 发送请求到上游
        send_request_func = self._get_provider(channel, official_key)
        upstream_response = await send_request_func(model_name, converted_body)

        # 8. 处理响应
        if is_stream:
            if upstream_response.status_code >= 400:
                error_content = await upstream_response.aread()
                return JSONResponse(
                    status_code=upstream_response.status_code,
                    content={"error": {"message": error_content.decode(), "type": "upstream_error"}}
                )
            return StreamingResponse(self._stream_response(upstream_response, target_format, model_name, preset.id if preset else None), media_type="text/event-stream")
        else:
            return await self._handle_non_stream_response(upstream_response, target_format, model_name, preset.id if preset else None)

    async def _stream_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int]):
        rules = []
        if preset_id:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset_id, PresetRegexRule.type == "post")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()

        async for chunk in upstream_response.aiter_bytes():
            if chunk.startswith(b'data: '):
                try:
                    chunk_data = json.loads(chunk[6:])
                    converted_chunk = self._convert_chunk(chunk_data, upstream_format, model)
                    
                    if rules and "choices" in converted_chunk:
                        for choice in converted_chunk["choices"]:
                            if "delta" in choice and "content" in choice["delta"]:
                                choice["delta"]["content"] = regex_service.process(choice["delta"]["content"], rules)

                    yield f"data: {json.dumps(converted_chunk)}\n\n"
                except json.JSONDecodeError:
                    continue
        yield "data: [DONE]\n\n"

    async def _handle_non_stream_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int]):
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(
                status_code=upstream_response.status_code,
                content={"error": {"message": error_content.decode(), "type": "upstream_error"}}
            )

        response_json = upstream_response.json()
        converted_response = self._convert_response(response_json, upstream_format, model)
        if preset_id:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset_id, PresetRegexRule.type == "post")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()

            if rules and "choices" in converted_response:
                for choice in converted_response["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        choice["message"]["content"] = regex_service.process(choice["message"]["content"], rules)

        return JSONResponse(content=converted_response)

    def _get_provider(self, channel: Channel, official_key: OfficialKey):
        if channel.type == "openai":
            provider = openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body: provider.create_chat_completion(body)
        elif channel.type == "gemini":
            provider = gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body: provider.generate_content(model, body)
        raise ValueError(f"Unsupported channel type: {channel.type}")

    def _convert_chunk(self, chunk: dict, upstream_format: str, model: str):
        if upstream_format == self.incoming_format:
            return chunk

        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_stream_chunk(chunk, model)

        # OpenAI -> Gemini conversion is not implemented for streaming in this direction
        return chunk

    def _convert_response(self, response: dict, upstream_format: str, model: str):
        if upstream_format == self.incoming_format:
            return response

        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_response(response, model)

        # OpenAI -> Gemini conversion is not implemented for non-streaming in this direction
        return response
