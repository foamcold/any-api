from typing import Dict, Any, Tuple, List
import json
import time
import uuid

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
    contents = []
    
    # 1. 合并和处理消息
    all_messages = preset_messages + openai_request.get("messages", [])
    merged_messages = _merge_messages(all_messages)
    
    # 2. 分离 system 指令
    system_parts = []
    other_messages = []
    for msg in merged_messages:
        if msg["role"] == "system":
            system_parts.append({"text": msg["content"]})
        else:
            other_messages.append(msg)

    # 3. 转换为 Gemini contents 格式
    for msg in other_messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # 4. 构建最终 payload
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": openai_request.get("temperature", 0.7),
            "topP": openai_request.get("top_p", 1.0),
            "maxOutputTokens": openai_request.get("max_tokens", 2048),
        }
    }
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}
        
    model_name = openai_request.get("model", "gemini-1.5-pro")
    
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