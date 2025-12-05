import logging
import uuid
import json
import asyncio
import time
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any

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
from app.services.token_service import token_service
from app.services.log_and_stat_service import update_key_and_log_usage
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

        start_time = time.time()
        
        try:
            channel = self.exclusive_key_obj.channel
            if not channel:
                self.logger.error(f"[{self.request_id}] 专属密钥 {self.exclusive_key_obj.id} 未绑定渠道。")
                return JSONResponse(status_code=400, content={"error": "Exclusive key is not associated with a channel."})
            
            official_key = await gemini_service.get_active_key(self.db, channel.id)
            if not official_key:
                self.logger.error(f"[{self.request_id}] 渠道 {channel.id} 无可用官方密钥。")
                return JSONResponse(status_code=500, content={"error": "No active official key available for the channel."})

            preset = self.exclusive_key_obj.preset
            preset_id = preset.id if preset else None

            # 预处理用户输入
            processed_body = await self._preprocess_request_body(body, preset_id)
            
            # 提取预设消息
            preset_messages, preset_model = await self._get_preset_messages(preset)

            target_format = channel.type
            channel_model = channel.model
            
            # 转换请求体，并获取最终的输入消息列表用于计算Token
            converted_body, final_input_messages, model_name = self._convert_request(
                processed_body, preset_messages, target_format, preset_model, channel_model
            )
            
            send_request_func = self._get_provider_for_chat(channel, official_key)
            
            self.logger.debug(f"[{self.request_id}] 准备发送请求到上游。模型: {model_name}, 流式: {is_upstream_stream}")
            upstream_response = await send_request_func(model_name, converted_body, is_upstream_stream)
            
            latency = time.time() - start_time

            if upstream_response.status_code >= 400:
                error_content = await upstream_response.aread()
                await self._log_and_update(official_key, model_name, final_input_messages, "", latency, upstream_response.status_code, "error")
                return JSONResponse(status_code=upstream_response.status_code, content=json.loads(error_content))

            if is_upstream_stream:
                self.logger.debug(f"[{self.request_id}] 开始处理上游流式响应。")
                return StreamingResponse(
                    self._stream_response(upstream_response, target_format, model_name, preset_id, final_input_messages, start_time, official_key),
                    media_type="text/event-stream"
                )
            else:
                self.logger.debug(f"[{self.request_id}] 开始处理上游非流式响应。")
                response_json = upstream_response.json()
                return await self._handle_non_stream_response(response_json, target_format, model_name, preset_id, final_input_messages, latency, official_key)

        except Exception as e:
            self.logger.error(f"[{self.request_id}] 预设代理请求处理时发生意外错误: {e}", exc_info=True)
            latency = time.time() - start_time
            # 尝试获取 official_key 以记录错误
            official_key = None
            try:
                channel = self.exclusive_key_obj.channel
                if channel:
                    official_key = await gemini_service.get_active_key(self.db, channel.id)
            except Exception:
                pass # 忽略获取密钥时的错误
            
            if official_key:
                 await self._log_and_update(official_key, original_model, body.get("messages", []), "", latency, 500, "error")

            return JSONResponse(status_code=500, content={"error": "An unexpected error occurred in PresetProxyService."})

    async def _stream_response(self, upstream_response, upstream_format: str, model: str, preset_id: Optional[int], final_input_messages: list, start_time: float, official_key: OfficialKey):
        ttft = 0.0
        first_token_received = False
        full_output = []
        buffer = b""
        
        self.logger.debug(f"[{self.request_id}] 流式响应转换开始。")

        async for chunk in upstream_response.aiter_bytes():
            if not first_token_received and chunk.strip():
                ttft = time.time() - start_time
                first_token_received = True
                self.logger.debug(f"[{self.request_id}] 收到首个Token，TTFT: {ttft:.4f}s")

            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line: continue
                if line == b"data: [DONE]":
                    self.logger.debug(f"[{self.request_id}] 收到 [DONE] 信号。")
                    yield "data: [DONE]\n\n"
                    
                    # 流结束后记录日志
                    latency = time.time() - start_time
                    output_text = "".join(full_output)
                    await self._log_and_update(official_key, model, final_input_messages, output_text, latency, 200, "ok", ttft, True)
                    return
                
                if line.startswith(b"data: "):
                    try:
                        chunk_data = json.loads(line[6:])
                        converted_chunk = self._convert_chunk(chunk_data, upstream_format, model)
                        if not converted_chunk: continue

                        # 后处理并累加输出
                        processed_text = await self._postprocess_and_accumulate_chunk(converted_chunk, preset_id)
                        if processed_text:
                            full_output.append(processed_text)

                        yield f"data: {json.dumps(converted_chunk)}\n\n"
                    except json.JSONDecodeError:
                        self.logger.warning(f"[{self.request_id}] JSON解码流式块失败: {line}")
                        continue
        
        # 确保即使没有 [DONE] 信号也能记录日志
        self.logger.debug(f"[{self.request_id}] 流式响应迭代结束。")
        latency = time.time() - start_time
        output_text = "".join(full_output)
        await self._log_and_update(official_key, model, final_input_messages, output_text, latency, 200, "ok", ttft, True)
        yield "data: [DONE]\n\n"

    async def _handle_non_stream_response(self, response_json: dict, upstream_format: str, model: str, preset_id: Optional[int], final_input_messages: list, latency: float, official_key: OfficialKey):
        self.logger.debug(f"[{self.request_id}] 非流式响应转换开始。")
        converted_response = self._convert_response(response_json, upstream_format, model, is_stream=False)
        
        output_text = ""
        # 后处理并提取输出文本
        if "choices" in converted_response: # OpenAI 格式
            for choice in converted_response["choices"]:
                if "message" in choice and "content" in choice["message"] and choice["message"]["content"]:
                    processed_content = await self._apply_all_rules(choice["message"]["content"], "post", preset_id)
                    choice["message"]["content"] = processed_content
                    output_text += processed_content
        elif "candidates" in converted_response: # Gemini 格式
            for candidate in converted_response["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            processed_text = await self._apply_all_rules(part["text"], "post", preset_id)
                            part["text"] = processed_text
                            output_text += processed_text
        
        # 记录日志
        await self._log_and_update(official_key, model, final_input_messages, output_text, latency, 200, "ok")

        self.logger.debug(f"[{self.request_id}] 非流式响应处理完成，返回结果。")
        return JSONResponse(content=converted_response)

    async def _preprocess_request_body(self, body: dict, preset_id: Optional[int]) -> dict:
        """对传入的请求体进行预处理（正则、变量等）。"""
        self.logger.debug(f"[{self.request_id}] 开始预处理请求体。")
        # 创建 body 的深拷贝以避免修改原始字典
        processed_body = json.loads(json.dumps(body))

        if self.incoming_format == "openai":
            for msg in processed_body.get("messages", []):
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = await self._apply_all_rules(msg["content"], "pre", preset_id)
        elif self.incoming_format == "gemini":
            for content in processed_body.get("contents", []):
                for part in content.get("parts", []):
                    if "text" in part:
                        part["text"] = await self._apply_all_rules(part["text"], "pre", preset_id)
        
        self.logger.debug(f"[{self.request_id}] 请求体预处理完成。")
        return processed_body

    async def _get_preset_messages(self, preset: Optional[Preset]) -> tuple[list, Optional[str]]:
        """获取并处理预设消息。"""
        if not preset:
            return [], None

        self.logger.debug(f"[{self.request_id}] 找到关联预设: {preset.name} (ID: {preset.id})")
        messages_data = json.loads(preset.content)
        preset_model = messages_data.get("model") if isinstance(messages_data, dict) else None
        
        if isinstance(messages_data, dict) and 'preset' in messages_data:
            messages = messages_data['preset']
        elif isinstance(messages_data, dict) and 'messages' in messages_data:
            messages = messages_data['messages']
        elif isinstance(messages_data, list):
            messages = messages_data
        else:
            messages = []

        # 对预设内容本身也应用变量处理
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                original_content = msg["content"]
                processed_content = variable_service.parse_variables(original_content)
                if original_content != processed_content:
                    self.logger.debug(f"[{self.request_id}] 预设内容变量处理: '{original_content}' -> '{processed_content}'")
                    msg["content"] = processed_content
        
        return messages, preset_model

    def _convert_request(self, body: dict, preset_messages: list, target_format: str, preset_model: Optional[str] = None, channel_model: Optional[str] = None) -> tuple[dict, list, str]:
        """转换请求体，并返回最终用于计算Token的输入消息列表。"""
        self.logger.debug(f"[{self.request_id}] 开始转换请求格式: {self.incoming_format} -> {target_format}")
        model_name = body.get("model")
        final_input_messages = []

        # 统一将输入转换为 OpenAI 的 'messages' 格式，以便后续处理和计算Token
        if self.incoming_format == "openai":
            user_messages = body.get("messages", [])
        elif self.incoming_format == "gemini":
            user_messages = []
            for msg in body.get("contents", []):
                text_content = "".join(p.get("text", "") for p in msg.get("parts", []))
                # 转换角色: gemini的'model'角色对应openai的'assistant'
                role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
                user_messages.append({"role": role, "content": text_content})
        else:
            user_messages = []

        # 合并预设消息和用户消息
        all_messages = preset_messages + user_messages
        final_input_messages = _merge_messages(all_messages)

        # --- 现在基于最终的 `final_input_messages` 来构建目标格式的请求体 ---

        converted_body = {}
        if target_format == "openai":
            converted_body = body # 复制原始请求的大部分结构
            converted_body["messages"] = final_input_messages
            # 移除 Gemini 特有的字段
            converted_body.pop("contents", None)
            converted_body.pop("generationConfig", None)
            
        elif target_format == "gemini":
            # 将合并后的消息列表转换为 Gemini 的 'contents' 格式
            contents = []
            for msg in final_input_messages:
                # 转换角色: openai的'assistant'角色对应gemini的'model'
                role = "model" if msg.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
            
            # 继承原始请求的 generationConfig
            generation_config = body.get("generationConfig", {})
            
            converted_body = {
                "contents": contents,
                "generationConfig": generation_config
            }
        else: # 无需转换
            converted_body = body

        self.logger.debug(f"[{self.request_id}] 请求格式转换完成。")
        return converted_body, final_input_messages, model_name

    async def _postprocess_and_accumulate_chunk(self, chunk: dict, preset_id: Optional[int]) -> Optional[str]:
        """对流式块进行后处理，并返回提取的文本内容用于累加。"""
        extracted_text = None
        if "choices" in chunk: # OpenAI 格式
            for choice in chunk["choices"]:
                if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                    original_text = choice["delta"]["content"]
                    processed_text = await self._apply_all_rules(original_text, "post", preset_id)
                    choice["delta"]["content"] = processed_text
                    extracted_text = processed_text
        elif "candidates" in chunk: # Gemini 格式
            for candidate in chunk["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            original_text = part["text"]
                            processed_text = await self._apply_all_rules(original_text, "post", preset_id)
                            part["text"] = processed_text
                            extracted_text = processed_text
        return extracted_text

    async def _log_and_update(self, official_key: OfficialKey, model: str, final_input_messages: list, output_text: str, latency: float, status_code: int, status: str, ttft: float = 0.0, is_stream: bool = False):
        """统一的日志记录和统计更新入口。"""
        self.logger.debug(f"[{self.request_id}] 准备记录日志和更新统计。状态: {status}, 状态码: {status_code}")
        try:
            input_tokens = token_service.count_input_tokens(final_input_messages, model)
            output_tokens = token_service.count_output_tokens(output_text, model)
            
            self.logger.debug(f"[{self.request_id}] Token 计算完成。输入: {input_tokens}, 输出: {output_tokens}")

            await update_key_and_log_usage(
                db=self.db,
                key_obj=official_key, # 在预设模式下，我们追踪的是官方密钥的消耗
                user=self.user,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status_code=status_code,
                latency=latency,
                ttft=ttft,
                is_stream=is_stream,
                status=status
            )
        except Exception as e:
            self.logger.error(f"[{self.request_id}] 在 _log_and_update 中发生错误: {e}", exc_info=True)

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
        preset_id = preset.id if preset else None

        # --- 1. 预处理 ---
        if self.incoming_format == "openai":
            for msg in body.get("messages", []):
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = await self._apply_all_rules(msg["content"], "pre", preset_id)
        elif self.incoming_format == "gemini":
            for content in body.get("contents", []):
                for part in content.get("parts", []):
                    if "text" in part:
                        part["text"] = await self._apply_all_rules(part["text"], "pre", preset_id)
        
        preset_messages = []
        if preset:
            messages_data = json.loads(preset.content)
            messages = messages_data.get('preset', messages_data.get('messages', [])) if isinstance(messages_data, dict) else messages_data if isinstance(messages_data, list) else []
            for msg in messages:
                if "content" in msg and isinstance(msg["content"], str):
                    original_content = msg["content"]
                    processed_content = variable_service.parse_variables(original_content)
                    if original_content != processed_content:
                        self.logger.debug(f"[{self.request_id}] 预设内容变量处理: '{original_content}' -> '{processed_content}'")
                        msg["content"] = processed_content
            preset_messages = messages

        converted_body, final_input_messages, model_name = self._convert_request(body, preset_messages, target_format)

        # --- 2. 请求上游 ---
        send_request_func = self._get_provider_for_chat(channel, official_key)
        start_time = time.time()
        upstream_task = asyncio.create_task(send_request_func(model_name, converted_body, False))

        # --- 3. 发送心跳包 ---
        while not upstream_task.done():
            self.logger.debug(f"[{self.request_id}] 伪流: 发送心跳包。")
            heartbeat_chunk = {}
            if self.incoming_format == "openai":
                heartbeat_chunk = {"id": f"chatcmpl-heartbeat-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(asyncio.get_event_loop().time()), "model": original_model, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
            elif self.incoming_format == "gemini":
                heartbeat_chunk = {"candidates": [{"content": {"parts": [{"text": ""}], "role": "model"}, "finishReason": None, "index": 0}]}
            if heartbeat_chunk:
                yield f"data: {json.dumps(heartbeat_chunk)}\n\n"
            await asyncio.sleep(1)

        upstream_response = await upstream_task
        latency = time.time() - start_time
        
        if upstream_response.status_code >= 400:
            error_content = await upstream_response.aread()
            error_payload = {"error": json.loads(error_content)}
            await self._log_and_update(official_key, model_name, final_input_messages, "", latency, upstream_response.status_code, "error", is_stream=True)
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # --- 4. 后处理和日志记录 ---
        response_json = upstream_response.json()
        final_response = self._convert_response(response_json, target_format, original_model, is_stream=False)
        
        output_text = ""
        if "choices" in final_response: # OpenAI 格式
            for choice in final_response["choices"]:
                if "message" in choice and "content" in choice["message"] and choice["message"]["content"]:
                    processed_content = await self._apply_all_rules(choice["message"]["content"], "post", preset_id)
                    choice["message"]["content"] = processed_content
                    output_text += processed_content
        elif "candidates" in final_response: # Gemini 格式
            for candidate in final_response["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            processed_text = await self._apply_all_rules(part["text"], "post", preset_id)
                            part["text"] = processed_text
                            output_text += processed_text
        
        await self._log_and_update(official_key, model_name, final_input_messages, output_text, latency, 200, "ok", is_stream=True)

        # --- 5. 模拟流式返回 ---
        self.logger.debug(f"[{self.request_id}] 伪流: 将完整响应包装为流式块。")
        if self.incoming_format == "openai":
            for i, choice in enumerate(final_response.get("choices", [])):
                chunk = {"id": final_response.get("id"), "object": "chat.completion.chunk", "created": final_response.get("created"), "model": final_response.get("model"), "choices": [{"index": i, "delta": {"content": choice.get("message", {}).get("content", "")}, "finish_reason": choice.get("finish_reason")}]}
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
             yield f"data: {json.dumps(final_response)}\n\n"

        yield "data: [DONE]\n\n"
