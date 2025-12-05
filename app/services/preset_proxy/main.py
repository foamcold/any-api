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
from app.services.preset_proxy.adapters import openai_adapter, gemini_adapter
from app.services.preset_proxy.providers import openai_provider, gemini_provider
from app.services.variable_service import variable_service
from app.services.gemini_service import gemini_service
from app.services.regex_service import regex_service

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
        """
        统一处理预设模式代理请求的核心方法。
        """
        self.logger.debug(f"[{self.request_id}] 接收到新的预设模式请求。")

        # 1. 获取渠道和官方密钥
        channel = self.exclusive_key_obj.channel
        if not channel:
            raise Exception("专属密钥未绑定渠道")
        
        # 此处需要一个服务来获取活跃的官方密钥
        official_key = await gemini_service.get_active_key(self.db, channel.id)


        # 2. 获取预设并处理变量
        preset = self.exclusive_key_obj.preset
        preset_messages = []
        if preset:
            # The preset content is a JSON string of a list of messages
            messages_data = json.loads(preset.content)
            
            # 支持两种格式: [{"role":...}] 或 {"messages": [{"role":...}]}
            if isinstance(messages_data, dict) and 'messages' in messages_data:
                messages = messages_data['messages']
            elif isinstance(messages_data, list):
                messages = messages_data
            else:
                messages = []

            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = variable_service.parse_variables(msg["content"])
            preset_messages = messages

        # 3. 解析请求体并应用前置正则
        body = await request.json()
        if preset:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset.id, PresetRegexRule.type == "pre")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()
            
            for msg in body.get("messages", []):
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = regex_service.process(msg["content"], rules)

        # 4. 格式转换和预设注入
        target_format = channel.type
        self.logger.debug(f"[{self.request_id}] Target channel format: {target_format}")
        converted_body, model_name = self._convert_request(body, preset_messages, target_format)

        # 5. 发送请求到上游
        send_request_func = self._get_provider(channel, official_key)
        upstream_response = await send_request_func(model_name, converted_body)

        # 6. 处理响应
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            is_stream = body.get("stream", False)
            
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
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            # Here you might want to convert the error to the client's expected format
            yield f"data: {json.dumps({'error': {'message': error_content.decode(), 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            # Ensure the stream is closed
            await upstream_response.aclose()
            return

        rules = []
        if preset_id:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset_id, PresetRegexRule.type == "post")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()

        buffer = b""
        async for chunk in upstream_response.aiter_bytes():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()

                if not line:
                    continue

                self.logger.debug(f"[{self.request_id}] 处理行: {line}")

                if line == b"data: [DONE]":
                    self.logger.debug(f"[{self.request_id}] 检测到上游流结束标志 [DONE]")
                    yield "data: [DONE]\n\n"
                    return

                if line.startswith(b"data: "):
                    try:
                        chunk_data = json.loads(line[6:])
                        self.logger.debug(f"[{self.request_id}] 解析后的JSON数据: {chunk_data}")
                        converted_chunk = self._convert_chunk(chunk_data, upstream_format, model)

                        if not converted_chunk:
                            self.logger.debug(f"[{self.request_id}] 转换后的数据块为空，跳过。")
                            continue

                        self.logger.debug(f"[{self.request_id}] 转换后的数据块: {converted_chunk}")

                        # Apply regex to content if it exists
                        if rules and "candidates" in converted_chunk:
                            for candidate in converted_chunk["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part in candidate["content"]["parts"]:
                                        if "text" in part:
                                            part["text"] = regex_service.process(part["text"], rules)

                        final_chunk_str = f"data: {json.dumps(converted_chunk)}\n\n"
                        self.logger.debug(f"[{self.request_id}] 准备发送给客户端的数据: {final_chunk_str.strip()}")
                        yield final_chunk_str
                    except json.JSONDecodeError:
                        self.logger.warning(f"[{self.request_id}] JSON解析失败，跳过该行: {line}")
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

    def _convert_request(self, body: dict, preset_messages: list, target_format: str):
        # No conversion needed if formats match
        if self.incoming_format == target_format:
            self.logger.debug(f"[{self.request_id}] Request format matches channel format ({self.incoming_format}), no conversion needed.")
            return body, body.get("model")

        conversion_direction = f"{self.incoming_format} -> {target_format}"
        self.logger.debug(f"[{self.request_id}] Converting request body: {conversion_direction}")

        # OpenAI (client) -> Gemini (channel)
        if self.incoming_format == "openai" and target_format == "gemini":
            return openai_adapter.to_gemini_request(body, preset_messages)
        
        # Gemini (client) -> OpenAI (channel)
        if self.incoming_format == "gemini" and target_format == "openai":
            if self.is_stream_override is not None:
                is_stream = self.is_stream_override
            else:
                is_stream = body.get("stream", False)
            return gemini_adapter.to_openai_request(body, preset_messages, is_stream)

        # Fallback for unhandled conversions
        self.logger.warning(f"[{self.request_id}] Unhandled request conversion: {conversion_direction}")
        return body, body.get("model")

    def _get_provider(self, channel: Channel, official_key: OfficialKey):
        if channel.type == "openai":
            provider = openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body: provider.create_chat_completion(body)
        elif channel.type == "gemini":
            provider = gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body: provider.generate_content(model, body)
        raise ValueError(f"Unsupported channel type: {channel.type}")

    def _convert_chunk(self, chunk: dict, upstream_format: str, model: str):
        # No conversion needed
        if upstream_format == self.incoming_format:
            return chunk

        conversion_direction = f"{upstream_format} -> {self.incoming_format}"
        if not hasattr(self, '_logged_chunk_conversion'):
            self.logger.debug(f"[{self.request_id}] Converting stream chunks: {conversion_direction}")
            self._logged_chunk_conversion = True

        # Gemini (upstream) -> OpenAI (client)
        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_stream_chunk(chunk, model)

        # OpenAI (upstream) -> Gemini (client)
        if upstream_format == "openai" and self.incoming_format == "gemini":
            return gemini_adapter.from_openai_stream_chunk(chunk)

        if not hasattr(self, '_logged_chunk_conversion_warning'):
            self.logger.warning(f"[{self.request_id}] Unhandled chunk conversion: {conversion_direction}")
            self._logged_chunk_conversion_warning = True
        return chunk

    def _convert_response(self, response: dict, upstream_format: str, model: str):
        # No conversion needed
        if upstream_format == self.incoming_format:
            self.logger.debug(f"[{self.request_id}] Response format matches client format ({self.incoming_format}), no conversion needed.")
            return response

        conversion_direction = f"{upstream_format} -> {self.incoming_format}"
        self.logger.debug(f"[{self.request_id}] Converting non-stream response: {conversion_direction}")

        # Gemini (upstream) -> OpenAI (client)
        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_response(response, model)

        # OpenAI (upstream) -> Gemini (client)
        if upstream_format == "openai" and self.incoming_format == "gemini":
            return gemini_adapter.from_openai_response(response)

        self.logger.warning(f"[{self.request_id}] Unhandled response conversion: {conversion_direction}")
        return response
