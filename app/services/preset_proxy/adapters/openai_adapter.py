from typing import Dict, Any, Tuple, List
import json
import time
import uuid
import logging

logger = logging.getLogger(__name__)

def _merge_messages(messages: list) -> list:
    """
    合并连续的、角色相同的消息。
    """
    if not messages:
        return []

    merged = [messages[0]]
    for i in range(1, len(messages)):
        current_msg = messages[i]
        last_msg = merged[-1]

        if current_msg.get("role") == last_msg.get("role"):
            if "content" in last_msg and isinstance(last_msg["content"], str):
                last_msg["content"] += "\n" + current_msg.get("content", "")
        else:
            merged.append(current_msg)
    
    return merged

def to_gemini_request(openai_request: Dict[str, Any], preset_messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str]:
    """
    将 OpenAI 请求格式转换为 Gemini 请求格式，并注入预设。
    """
    logger.debug(f"适配器接收到预设消息: {preset_messages}")
    logger.debug(f"适配器接收到用户消息: {openai_request.get('messages', [])}")

    # 1. 智能合并消息，将用户消息替换 'user_input' 占位符
    all_messages = []
    user_messages = openai_request.get("messages", [])
    user_input_found = False

    for msg in preset_messages:
        if msg.get("type") == "user_input":
            all_messages.extend(user_messages)
            user_input_found = True
        else:
            all_messages.append(msg)

    if not user_input_found:
        all_messages.extend(user_messages)

    logger.debug(f"合并后的所有消息: {all_messages}")
    
    # 2. 合并连续的同角色消息
    merged_messages = _merge_messages(all_messages)
    logger.debug(f"处理连续角色后的消息: {merged_messages}")

    # 3. 分离 system 指令
    system_parts = []
    user_model_messages = []
    for msg in merged_messages:
        if msg["role"] == "system":
            system_parts.append({"text": msg["content"]})
        else:
            user_model_messages.append(msg)
    logger.debug(f"分离出的系统指令: {system_parts}")
    logger.debug(f"分离出的用户/模型消息: {user_model_messages}")

    # 4. 对剩余消息进行二次合并，以处理 [user, system, user] -> [user, user] 的情况
    final_user_model_messages = _merge_messages(user_model_messages)
    logger.debug(f"二次合并后的最终用户/模型消息: {final_user_model_messages}")

    # 5. 转换为 Gemini contents 格式，并将系统指令注入到第一条用户消息中
    contents = []
    combined_system_text = "\n".join(p.get("text", "") for p in system_parts if p.get("text")).strip()
    logger.debug(f"合并后的系统指令文本: '{combined_system_text}'")

    for i, msg in enumerate(final_user_model_messages):
        role = "user" if msg["role"] == "user" else "model"
        current_content = msg.get("content", "")

        if i == 0 and role == "user" and combined_system_text:
            final_content = f"{combined_system_text}\n\n{current_content}"
            contents.append({"role": role, "parts": [{"text": final_content}]})
        else:
            contents.append({"role": role, "parts": [{"text": current_content}]})
    
    logger.debug(f"转换为 Gemini contents 的最终内容: {contents}")

    # 6. 构建最终 payload，不再使用 system_instruction
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": openai_request.get("temperature", 0.7),
            "topP": openai_request.get("top_p", 1.0),
            "maxOutputTokens": openai_request.get("max_tokens", 2048),
        }
    }
    
    model_name = openai_request.get("model", "gemini-1.5-pro")
    logger.debug(f"最终发送给上游的 payload: {payload}")
    
    return payload, model_name

def from_gemini_response(gemini_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    将 Gemini 响应格式转换为 OpenAI 响应格式。
    """
    choices = []
    candidates = gemini_response.get("candidates", [])
    
    for i, candidate in enumerate(candidates):
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text_content = "".join([p.get("text", "") for p in parts])
        
        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": candidate.get("finishReason", "stop"),
        })
        
    usage = gemini_response.get("usageMetadata", {})
    return {
        "id": f"chatcmpl-gemini-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }

def from_gemini_stream_chunk(chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    将 Gemini 流式块转换为 OpenAI 流式块。
    """
    choices = []
    if "candidates" in chunk:
        for i, candidate in enumerate(chunk["candidates"]):
            delta = {}
            finish_reason = None
            
            if "content" in candidate and "parts" in candidate["content"]:
                content_str = "".join([p.get("text", "") for p in candidate["content"]["parts"]])
                if content_str:
                    delta["content"] = content_str
            
            if candidate.get("finishReason"):
                finish_reason = candidate.get("finishReason")
                if finish_reason == "MAX_TOKENS":
                    finish_reason = "length"
            
            choices.append({"index": i, "delta": delta, "finish_reason": finish_reason})
            
    return {
        "id": f"chatcmpl-gemini-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }