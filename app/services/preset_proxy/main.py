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
from app.services.llm_proxy.adapters import openai_adapter, gemini_adapter
from app.services.preset_proxy.providers import openai_provider, gemini_provider
from app.services.preset_proxy.utils import _merge_messages
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
            self.logger.debug(f"[{self.request_id}] 找到关联预设: {preset.name} (ID: {preset.id})")
            self.logger.debug(f"[{self.request_id}] 原始预设内容: {preset.content}")
            # The preset content is a JSON string of a list of messages
            messages_data = json.loads(preset.content)
            
            # 支持两种格式: [{"role":...}] 或 {"messages": [{"role":...}]}
            if isinstance(messages_data, dict) and 'preset' in messages_data:
                messages = messages_data['preset']
            elif isinstance(messages_data, dict) and 'messages' in messages_data:
                messages = messages_data['messages']
            elif isinstance(messages_data, list):
                messages = messages_data
            else:
                messages = []
            self.logger.debug(f"[{self.request_id}] 解析后的消息列表: {messages}")

            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    original_content = msg["content"]
                    msg["content"] = variable_service.parse_variables(original_content)
                    if original_content != msg["content"]:
                        self.logger.debug(f"[{self.request_id}] 预设变量替换: '{original_content}' -> '{msg['content']}'")
            
            preset_messages = messages
            self.logger.debug(f"[{self.request_id}] 最终准备传递给适配器的预设消息: {preset_messages}")
        else:
            self.logger.debug(f"[{self.request_id}] 未找到关联预设。")

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
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            is_stream = body.get("stream", False)
            
        send_request_func = self._get_provider_for_chat(channel, official_key)
        upstream_response = await send_request_func(model_name, converted_body, is_stream)

        # 6. 处理响应
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            is_stream = body.get("stream", False)

        # 检查上游响应是否为流式
        is_upstream_stream = "text/event-stream" in upstream_response.headers.get("content-type", "")

        if is_stream:
            # 客户端期望流式响应
            if is_upstream_stream:
                # 完美情况：上游返回流式，直接代理
                self.logger.debug(f"[{self.request_id}] 客户端与上游均为流式，直接代理。")
                return StreamingResponse(self._stream_response(upstream_response, target_format, model_name, preset.id if preset else None), media_type="text/event-stream")
            else:
                # 不匹配：上游返回非流式，模拟流式响应
                self.logger.warning(f"[{self.request_id}] 响应类型不匹配：客户端期望流式，但上游返回非流式。启动流模拟。")
                return StreamingResponse(self._simulate_stream_from_full_response(upstream_response, target_format, model_name, preset.id if preset else None), media_type="text/event-stream")
        else:
            # 客户端期望非流式响应
            if is_upstream_stream:
                # 不匹配：上游返回流式，聚合成完整响应
                self.logger.warning(f"[{self.request_id}] 响应类型不匹配：客户端期望非流式，但上游返回流式。将聚合流式响应。")
                full_response = await self._aggregate_stream_response(upstream_response, target_format, model_name)
                return await self._handle_non_stream_response(full_response, target_format, model_name, preset.id if preset else None)
            else:
                # 完美情况：上游返回非流式，直接处理
                self.logger.debug(f"[{self.request_id}] 客户端与上游均为非流式，直接处理。")
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
        model_name = body.get("model")

        # 即使格式相同，也需要注入预设
        if self.incoming_format == target_format:
            self.logger.debug(f"[{self.request_id}] 格式匹配 ({self.incoming_format})，但仍需注入预设。")
            
            if self.incoming_format == "openai":
                # 合并 OpenAI 消息
                all_messages = preset_messages + body.get("messages", [])
                body["messages"] = self._merge_messages(all_messages)
                return body, model_name
            
            elif self.incoming_format == "gemini":
                # 当输入和输出都是 Gemini 时，应用统一的转换逻辑
                
                # 1. 规范化输入：将 gemini 的 'parts' 转换为 'content' 字符串
                normalized_incoming_messages = []
                for msg in body.get("contents", []):
                    text_content = "".join(p.get("text", "") for p in msg.get("parts", []))
                    normalized_incoming_messages.append({"role": msg.get("role"), "content": text_content})

                # 2. 合并预设和规范化后的消息，并将 'system' 转为 'user'
                all_messages = preset_messages + normalized_incoming_messages
                for msg in all_messages:
                    if msg.get("role") == "system":
                        msg["role"] = "user"
                
                # 3. 合并连续的消息
                merged_messages = _merge_messages(all_messages)
                
                # 4. 转换回 Gemini 'contents' 格式
                final_contents = []
                for msg in merged_messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    final_contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

                # 5. 更新 body
                body["contents"] = final_contents
                return body, model_name

        # --- 格式不同的情况 ---
        conversion_direction = f"{self.incoming_format} -> {target_format}"
        self.logger.debug(f"[{self.request_id}] 转换请求体: {conversion_direction}")

        if self.incoming_format == "openai" and target_format == "gemini":
            return openai_adapter.to_gemini_request(body, preset_messages)
        
        if self.incoming_format == "gemini" and target_format == "openai":
            is_stream = self.is_stream_override if self.is_stream_override is not None else body.get("stream", False)
            return gemini_adapter.to_openai_request(body, preset_messages, is_stream)

        self.logger.warning(f"[{self.request_id}] 未处理的转换: {conversion_direction}")
        return body, model_name

    def _get_provider_for_chat(self, channel: Channel, official_key: OfficialKey):
        if self.is_stream_override is not None:
            is_stream = self.is_stream_override
        else:
            # This is a bit of a hack, we need to get the body to check for stream,
            # but we don't have the request object here. We assume the body is already read in proxy_request
            # and this is just to determine the provider function signature.
            # A better solution would be to refactor how providers are called.
            is_stream = True # Default to stream for provider signature if not overriden

        if channel.type == "openai":
            provider = openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body, is_stream: provider.create_chat_completion(body)
        elif channel.type == "gemini":
            provider = gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
            return lambda model, body, is_stream: provider.generate_content(model, body, is_stream)
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

    async def _simulate_stream_from_full_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int]):
        """
        从一个完整的响应体中模拟流式响应。
        """
        # 确保我们能处理上游错误
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            yield f"data: {json.dumps({'error': {'message': error_content.decode(), 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        response_json = await upstream_response.json()
        self.logger.debug(f"[{self.request_id}] 模拟流 - 收到完整JSON: {response_json}")

        # 转换响应体
        converted_response = self._convert_response(response_json, upstream_format, model)
        self.logger.debug(f"[{self.request_id}] 模拟流 - 转换后的响应: {converted_response}")

        # 应用后置正则表达式
        if preset_id:
            from app.models.preset_regex import PresetRegexRule
            from sqlalchemy.future import select
            stmt = select(PresetRegexRule).filter(PresetRegexRule.preset_id == preset_id, PresetRegexRule.type == "post")
            result = await self.db.execute(stmt)
            rules = result.scalars().all()

            if rules:
                if "candidates" in converted_response: # Gemini format
                    for candidate in converted_response["candidates"]:
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    part["text"] = regex_service.process(part["text"], rules)
                elif "choices" in converted_response: # OpenAI format
                    for choice in converted_response["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            choice["message"]["content"] = regex_service.process(choice["message"]["content"], rules)
        
        self.logger.debug(f"[{self.request_id}] 模拟流 - 发送数据块: {converted_response}")
        yield f"data: {json.dumps(converted_response)}\n\n"
        self.logger.debug(f"[{self.request_id}] 模拟流 - 发送 [DONE] 标志")
        yield "data: [DONE]\n\n"

    async def _aggregate_stream_response(self, upstream_response, upstream_format: str, model: str) -> dict:
        """
        将流式响应聚合成一个完整的响应体。
        """
        full_response_content = ""
        # 注意：这里的 _stream_response 是一个生成器，我们需要迭代它来获取数据
        async for chunk_str in self._stream_response(upstream_response, upstream_format, model, None):
            if chunk_str.startswith("data:"):
                content = chunk_str[6:].strip()
                if content == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(content)
                    # 提取文本内容并拼接
                    if self.incoming_format == "openai": # Target is OpenAI
                        full_response_content += chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    elif self.incoming_format == "gemini": # Target is Gemini
                         full_response_content += chunk_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # 构建一个模拟的完整响应对象
        if self.incoming_format == "openai":
            return {"choices": [{"message": {"role": "assistant", "content": full_response_content}}]}
        elif self.incoming_format == "gemini":
            return {"candidates": [{"content": {"role": "model", "parts": [{"text": full_response_content}]}}]}
        return {}

    async def proxy_list_models(self, request: Request):
        """
        为预设模式代理模型列表请求，并传递查询参数。
        该方法现在将请求直接透传到与渠道关联的上游服务。
        """
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

        # 根据需要转换模型列表的格式
        target_format = channel.type
        if target_format == "gemini" and self.incoming_format == "openai":
            self.logger.debug(f"[{self.request_id}] 模型列表转换: Gemini -> OpenAI")
            from app.services.llm_proxy.adapters import openai_adapter # 复用LLM Proxy的adapter
            final_response = openai_adapter.from_gemini_to_openai_models(response_json)
        elif target_format == "openai" and self.incoming_format == "gemini":
            self.logger.debug(f"[{self.request_id}] 模型列表转换: OpenAI -> Gemini")
            from app.services.llm_proxy.adapters import gemini_adapter # 复用LLM Proxy的adapter
            final_response = gemini_adapter.from_openai_to_gemini_models(response_json)
        else:
            # 同构透传，无需转换
            final_response = response_json

        return JSONResponse(content=final_response)

    def _get_provider_for_models(self, channel: Channel, official_key: OfficialKey):
        """
        根据渠道类型为模型列表请求实例化并返回一个Provider。
        """
        if channel.type == "openai":
            return openai_provider.OpenAIProvider(api_key=official_key.key, base_url=channel.api_url)
        elif channel.type == "gemini":
            return gemini_provider.GeminiProvider(api_key=official_key.key, base_url=channel.api_url)
        raise ValueError(f"Unsupported channel type for listing models: {channel.type}")
