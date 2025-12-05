"""
负责处理传入的Gemini格式请求，并将其转换为其他上游提供商（如OpenAI）的格式。
同时，也负责将上游提供商的响应转换回Gemini格式。
"""
import json
from typing import Dict, Any, Tuple, Optional, List
from app.services.preset_proxy.utils import _merge_messages


def to_openai_request(gemini_request: Dict[str, Any], preset_messages: List[Dict[str, Any]], is_stream: bool, preset_model: Optional[str] = None, channel_model: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    将Gemini格式的请求（包含预设）转换为OpenAI格式。
    返回转换后的请求体和模型名称。
    """
    model_name = gemini_request.get("model")
    
    if preset_model:
        openai_model = preset_model
    elif channel_model and channel_model != "auto":
        openai_model = channel_model
    else:
        # 如果没有在预设或渠道中指定模型，则直接使用客户端传入的模型名称
        openai_model = model_name

    # 1. 将传入的Gemini消息（包括system_instruction和contents）规范化为简单的消息列表
    normalized_gemini_messages = []
    if "system_instruction" in gemini_request:
        system_text = gemini_request["system_instruction"].get("parts", [{}])[0].get("text", "")
        if system_text:
            normalized_gemini_messages.append({"role": "system", "content": system_text})

    for content in gemini_request.get("contents", []):
        role = "user" if content.get("role") == "user" else "assistant"
        # 注意：为了合并，这里简化了处理，只提取文本。这会丢失多模态信息。
        text_content = "".join(p.get("text", "") for p in content.get("parts", []))
        normalized_gemini_messages.append({"role": role, "content": text_content})

    # 2. 合并预设消息和规范化后的Gemini消息
    all_messages = (preset_messages or []) + normalized_gemini_messages
    merged_messages = _merge_messages(all_messages)

    # 3. 清理消息，只保留OpenAI API支持的字段
    final_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in merged_messages
    ]

    # 4. 构建OpenAI请求
    config = gemini_request.get("generationConfig", {})
    openai_request = {
        "model": openai_model,
        "messages": final_messages,
        "stream": is_stream,
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

def from_openai_to_gemini_models(openai_models_response: Dict[str, Any]) -> Dict[str, Any]:
   """
   将OpenAI格式的模型列表转换为Gemini格式。
   """
   gemini_models = []
   for model in openai_models_response.get("data", []):
       gemini_models.append({
           "name": f"models/{model.get('id')}",
           "version": "0.0.1", # Gemini需要此字段，但OpenAI不提供
           "displayName": model.get('id'),
           "description": f"Model {model.get('id')} provided by OpenAI.",
           "inputTokenLimit": 8192, # 默认值
           "outputTokenLimit": 2048, # 默认值
           "supportedGenerationMethods": [
               "generateContent",
               "streamGenerateContent"
           ],
       })
   
   return {
       "models": gemini_models
   }