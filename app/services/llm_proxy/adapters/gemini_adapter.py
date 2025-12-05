"""
负责处理传入的Gemini格式请求，并将其转换为其他上游提供商（如OpenAI）的格式。
同时，也负责将上游提供商的响应转换回Gemini格式。
"""
import json
from typing import Dict, Any, Tuple, Optional

def to_openai_request(gemini_request: Dict[str, Any], model_name: str) -> Tuple[Dict[str, Any], str]:
    """
    将Gemini格式的请求转换为OpenAI格式。
    返回转换后的请求体和原始模型名称。
    """
    # 简单的模型映射
    if "gemini-1.5-pro" in model_name:
        openai_model = "gpt-4-turbo"
    elif "gemini-1.5-flash" in model_name:
        openai_model = "gpt-3.5-turbo"
    else:
        openai_model = "gpt-4-turbo" # 默认

    # 转换 messages
    messages = []
    
    # 处理 system_instruction
    if "system_instruction" in gemini_request:
        system_text = gemini_request["system_instruction"].get("parts", [{}])[0].get("text", "")
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for content in gemini_request.get("contents", []):
        role = "user" if content.get("role") == "user" else "assistant"
        
        parts = content.get("parts", [])
        # OpenAI V API支持数组格式的content
        openai_content = []
        for part in parts:
            if "text" in part:
                openai_content.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part:
                inline_data = part["inlineData"]
                data_url = f"data:{inline_data['mimeType']};base64,{inline_data['data']}"
                openai_content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages.append({"role": role, "content": openai_content})

    # 转换 generationConfig
    config = gemini_request.get("generationConfig", {})
    openai_request = {
        "model": openai_model,
        "messages": messages,
        "temperature": config.get("temperature", 0.7),
        "top_p": config.get("topP", 1.0),
        "max_tokens": config.get("maxOutputTokens", 2048),
    }
    if "stopSequences" in config:
        openai_request["stop"] = config["stopSequences"]

    # 转换工具
    if "tools" in gemini_request:
        openai_tools = []
        for tool in gemini_request["tools"]:
            for func_dec in tool.get("functionDeclarations", []):
                openai_tools.append({
                    "type": "function",
                    "function": func_dec
                })
        openai_request["tools"] = openai_tools

    return openai_request, model_name

def from_openai_response(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    将OpenAI的（非流式）响应转换为Gemini格式。
    """
    candidates = []
    for choice in openai_response.get("choices", []):
        finish_reason = "STOP"
        raw_reason = choice.get("finish_reason")
        if raw_reason == "length":
            finish_reason = "MAX_TOKENS"
        elif raw_reason == "content_filter":
            finish_reason = "SAFETY"
        elif raw_reason == "tool_calls":
            finish_reason = "TOOL_CODE"

        # 转换 message content
        message = choice.get("message", {})
        parts = []
        if message.get("content"):
            parts.append({"text": message.get("content")})
        
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                parts.append({
                    "functionCall": {
                        "name": func.get("name"),
                        "args": json.loads(func.get("arguments", "{}"))
                    }
                })

        candidates.append({
            "content": {"parts": parts, "role": "model"},
            "finishReason": finish_reason,
            "index": choice.get("index"),
            "safetyRatings": [], # 默认为空
        })

    usage = openai_response.get("usage", {})
    return {
        "candidates": candidates,
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }
    }

def from_openai_stream_chunk(openai_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    将OpenAI的流式数据块转换为Gemini格式的流式数据块。
    """
    if not openai_chunk.get("choices"):
        return None

    choice = openai_chunk["choices"][0]
    delta = choice.get("delta", {})
    parts = []

    if "content" in delta and delta["content"]:
        parts.append({"text": delta["content"]})
    
    if "tool_calls" in delta:
        for tool_call in delta["tool_calls"]:
            func = tool_call.get("function", {})
            parts.append({
                "functionCall": {
                    "name": func.get("name"),
                    "args": json.loads(func.get("arguments", "{}"))
                }
            })

    # 如果没有实际内容，则不生成数据块
    if not parts:
        return None

    candidate = {
        "content": {"parts": parts, "role": "model"},
        "index": choice.get("index", 0),
    }

    # 处理结束原因
    if choice.get("finish_reason"):
        finish_reason = "STOP"
        raw_reason = choice.get("finish_reason")
        if raw_reason == "length": finish_reason = "MAX_TOKENS"
        elif raw_reason == "content_filter": finish_reason = "SAFETY"
        candidate["finishReason"] = finish_reason

    return {"candidates": [candidate]}