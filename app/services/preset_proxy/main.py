import logging
import uuid
import json
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.models.key import ExclusiveKey, OfficialKey
from app.models.user import User
from app.models.channel import Channel
from app.models.preset import Preset
from app.models.system_config import SystemConfig
from app.services.llm_proxy.adapters import openai_adapter, gemini_adapter
from app.services.preset_proxy.providers import openai_provider, gemini_provider
from app.services.preset_proxy.utils import _merge_messages
from app.services.variable_service import variable_service
from app.services.gemini_service import gemini_service
from app.services.regex_service import regex_service
from app.models.regex import RegexRule
from app.models.preset_regex import PresetRegexRule
from sqlalchemy.future import select


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

    async def _apply_all_rules(self, text: str, rule_type: str, preset_id: Optional[int]) -> str:
        original_text = text

        # 1. 全局正则
        if self.exclusive_key_obj.enable_regex:
            global_rules_stmt = select(RegexRule).filter(RegexRule.type == rule_type)
            global_rules_result = await self.db.execute(global_rules_stmt)
            global_rules = global_rules_result.scalars().all()
            if global_rules:
                processed_text = regex_service.process(text, global_rules)
                if processed_text != text:
                    self.logger.debug(f"[{self.request_id}] 全局正则({rule_type})应用: '{text}' -> '{processed_text}'")
                    text = processed_text

        # 2. 局部正则
        if preset_id:
            local_rules_stmt = select(PresetRegexRule).filter(
                PresetRegexRule.preset_id == preset_id,
                PresetRegexRule.type == rule_type
            )
            local_rules_result = await self.db.execute(local_rules_stmt)
            local_rules = local_rules_result.scalars().all()
            if local_rules:
                processed_text = regex_service.process(text, local_rules)
                if processed_text != text:
                    self.logger.debug(f"[{self.request_id}] 局部正则({rule_type})应用: '{text}' -> '{processed_text}'")
                    text = processed_text

        # 3. 变量处理 (默认开启)
        processed_text = variable_service.parse_variables(text)
        if processed_text != text:
            self.logger.debug(f"[{self.request_id}] 变量处理应用: '{text}' -> '{processed_text}'")
            text = processed_text
            
        return text

    async def proxy_request(self, body: dict, is_pseudo_stream: bool = False):
        """
        统一处理预设模式代理请求的核心方法。
        """
        self.logger.debug(f"[{self.request_id}] 接收到新的预设模式请求。")

        original_model = body.get("model", "unknown")
        client_wants_stream = body.get("stream", False)

        if self.is_stream_override is not None:
            is_upstream_stream = self.is_stream_override
        else:
            is_upstream_stream = client_wants_stream

        self.logger.debug(f"[{self.request_id}] 原始模型: {original_model}, 客户端期望流式: {client_wants_stream}, 上游将使用流式: {is_upstream_stream}")

        if is_pseudo_stream and client_wants_stream:
            self.logger.debug(f"[{self.request_id}] 检测到伪流请求，且客户端需要流式响应，启动模拟流响应。")
            return StreamingResponse(
                self._simulate_stream_response(body, original_model),
                media_type="text/event-stream"
            )

        channel = self.exclusive_key_obj.channel
        if not channel:
            raise Exception("专属密钥未绑定渠道")
        
        official_key = await gemini_service.get_active_key(self.db, channel.id)
        preset = self.exclusive_key_obj.preset
        preset_id = preset.id if preset else None

        # 预处理用户输入
        for msg in body.get("messages", []):
            if "content" in msg and isinstance(msg["content"], str):
                msg["content"] = await self._apply_all_rules(msg["content"], "pre", preset_id)

        preset_messages = []
        preset_model: Optional[str] = None
        if preset:
            self.logger.debug(f"[{self.request_id}] 找到关联预设: {preset.name} (ID: {preset.id})")
            messages_data = json.loads(preset.content)

            if isinstance(messages_data, dict):
                preset_model = messages_data.get("model")
            
            if isinstance(messages_data, dict) and 'preset' in messages_data:
                messages = messages_data['preset']
            elif isinstance(messages_data, dict) and 'messages' in messages_data:
                messages = messages_data['messages']
            elif isinstance(messages_data, list):
                messages = messages_data
            else:
                messages = []

            # 对预设内容本身也应用变量处理 (默认开启)
            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    original_content = msg["content"]
                    processed_content = variable_service.parse_variables(original_content)
                    if original_content != processed_content:
                        self.logger.debug(f"[{self.request_id}] 预设内容变量处理: '{original_content}' -> '{processed_content}'")
                        msg["content"] = processed_content
            
            preset_messages = messages

        target_format = channel.type
        channel_model = channel.model
        converted_body, model_name = self._convert_request(body, preset_messages, target_format, preset_model, channel_model)
        
        send_request_func = self._get_provider_for_chat(channel, official_key)
        upstream_response = await send_request_func(model_name, converted_body, is_upstream_stream)

        if is_upstream_stream:
            return StreamingResponse(self._stream_response(upstream_response, target_format, model_name, preset_id), media_type="text/event-stream")
        else:
            return await self._handle_non_stream_response(upstream_response, target_format, model_name, preset_id)

    async def _stream_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int]):
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            yield f"data: {json.dumps({'error': {'message': error_content.decode(), 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            await upstream_response.aclose()
            return

        buffer = b""
        async for chunk in upstream_response.aiter_bytes():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line: continue
                if line == b"data: [DONE]":
                    yield "data: [DONE]\n\n"
                    return
                if line.startswith(b"data: "):
                    try:
                        chunk_data = json.loads(line[6:])
                        converted_chunk = self._convert_chunk(chunk_data, upstream_format, model)
                        if not converted_chunk: continue

                        # OpenAI 格式后处理
                        if "choices" in converted_chunk:
                            for choice in converted_chunk["choices"]:
                                if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                    choice["delta"]["content"] = await self._apply_all_rules(choice["delta"]["content"], "post", preset_id)
                        
                        # Gemini 格式后处理
                        if "candidates" in converted_chunk:
                            for candidate in converted_chunk["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part in candidate["content"]["parts"]:
                                        if "text" in part:
                                            part["text"] = await self._apply_all_rules(part["text"], "post", preset_id)

                        yield f"data: {json.dumps(converted_chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
        yield "data: [DONE]\n\n"

    async def _handle_non_stream_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int]):
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(status_code=upstream_response.status_code, content={"error": {"message": error_content.decode(), "type": "upstream_error"}})

        response_json = upstream_response.json()
        converted_response = self._convert_response(response_json, upstream_format, model, is_stream=False)
        
        # 后处理
        if "choices" in converted_response:
            for choice in converted_response["choices"]:
                if "message" in choice and "content" in choice["message"] and choice["message"]["content"]:
                    choice["message"]["content"] = await self._apply_all_rules(choice["message"]["content"], "post", preset_id)

        return JSONResponse(content=converted_response)

    def _convert_request(self, body: dict, preset_messages: list, target_format: str, preset_model: Optional[str] = None, channel_model: Optional[str] = None):
        model_name = body.get("model")
        if self.incoming_format == target_format:
            if self.incoming_format == "openai":
                all_messages = preset_messages + body.get("messages", [])
                body["messages"] = _merge_messages(all_messages)
                return body, model_name
            elif self.incoming_format == "gemini":
                # 规范化传入的 Gemini 'contents' 为 'messages' 格式
                normalized_incoming_messages = []
                for msg in body.get("contents", []):
                    text_content = "".join(p.get("text", "") for p in msg.get("parts", []))
                    normalized_incoming_messages.append({"role": msg.get("role"), "content": text_content})

                # 合并预设消息和用户消息
                all_messages = preset_messages + normalized_incoming_messages
                
                # 将所有 system 消息的角色更改为 user
                for message in all_messages:
                    if message.get("role") == "system":
                        message["role"] = "user"

                # 合并相邻的 user 消息
                merged_messages = _merge_messages(all_messages)

                # 构建最终的 Gemini 'contents'
                final_contents = []
                for msg in merged_messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    final_contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                
                body["contents"] = final_contents
                
                # 确保移除 systemInstruction 字段
                if "systemInstruction" in body:
                    del body["systemInstruction"]
                
                # 移除不应存在于 Gemini 请求体中的字段
                if "model" in body:
                    del body["model"]
                if "stream" in body:
                    del body["stream"]

                return body, model_name
        if self.incoming_format == "openai" and target_format == "gemini":
            return openai_adapter.to_gemini_request(body, preset_messages)
        if self.incoming_format == "gemini" and target_format == "openai":
            is_stream = self.is_stream_override if self.is_stream_override is not None else body.get("stream", False)
            return gemini_adapter.to_openai_request(body, preset_messages, is_stream, preset_model, channel_model)
        return body, model_name

    def _get_provider_for_chat(self, channel: Channel, official_key: OfficialKey):
        if channel.type == "openai":
            provider = openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
            def openai_sender(model, body, is_stream):
                body['stream'] = is_stream
                return provider.create_chat_completion(body)
            return openai_sender
        elif channel.type == "gemini":
            provider = gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body, is_stream: provider.generate_content(model, body, is_stream)
        raise ValueError(f"Unsupported channel type: {channel.type}")

    def _convert_chunk(self, chunk: dict, upstream_format: str, model: str):
        if upstream_format == self.incoming_format: return chunk
        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_stream_chunk(chunk, model)
        if upstream_format == "openai" and self.incoming_format == "gemini":
            return gemini_adapter.from_openai_stream_chunk(chunk)
        return chunk

    def _convert_response(self, response: dict, upstream_format: str, model: str, is_stream: bool):
        if upstream_format == self.incoming_format: return response
        if upstream_format == "gemini" and self.incoming_format == "openai":
            return openai_adapter.from_gemini_response(response, model, is_stream)
        if upstream_format == "openai" and self.incoming_format == "gemini":
            return gemini_adapter.from_openai_response(response)
        return response

    async def proxy_list_models(self, request: Request, system_config: Optional[SystemConfig]):
        self.logger.debug(f"[{self.request_id}] 代理预设模式的模型列表请求（透传模式）。")
        params = request.query_params
        self.logger.debug(f"[{self.request_id}] 传递查询参数: {params}")
        channel = self.exclusive_key_obj.channel
        if not channel:
            return JSONResponse(status_code=404, content={"error": "Exclusive key is not associated with a channel."})
        official_key = await gemini_service.get_active_key(self.db, channel.id)
        if not official_key:
            return JSONResponse(status_code=500, content={"error": "No active official key available for the channel."})
        provider = self._get_provider_for_models(channel, official_key)
        upstream_response = await provider.list_models(params=params)
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            return JSONResponse(status_code=upstream_response.status_code, content=json.loads(error_content))
        response_json = upstream_response.json()
        target_format = channel.type
        if target_format == "gemini" and self.incoming_format == "openai":
            final_response = openai_adapter.from_gemini_to_openai_models(response_json)
        elif target_format == "openai" and self.incoming_format == "gemini":
            final_response = gemini_adapter.from_openai_to_gemini_models(response_json)
        else:
            final_response = response_json
        if system_config and system_config.pseudo_streaming_enabled:
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

    def _get_provider_for_models(self, channel: Channel, official_key: OfficialKey):
        if channel.type == "openai":
            return openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
        elif channel.type == "gemini":
            return gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
        raise ValueError(f"Unsupported channel type for listing models: {channel.type}")

    async def _simulate_stream_response(self, body: dict, original_model: str):
        channel = self.exclusive_key_obj.channel
        official_key = await gemini_service.get_active_key(self.db, channel.id)
        target_format = channel.type
        
        preset = self.exclusive_key_obj.preset
        preset_messages = []
        if preset:
            messages_data = json.loads(preset.content)
            messages = messages_data.get('preset', messages_data.get('messages', [])) if isinstance(messages_data, dict) else messages_data if isinstance(messages_data, list) else []
            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = variable_service.parse_variables(msg["content"])
            preset_messages = messages

        converted_body, model_name = self._convert_request(body, preset_messages, target_format)

        send_request_func = self._get_provider_for_chat(channel, official_key)
        upstream_task = asyncio.create_task(send_request_func(model_name, converted_body, False))

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
        final_response = self._convert_response(response_json, target_format, original_model, is_stream=False)
        
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
