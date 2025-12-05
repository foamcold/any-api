"""
负责处理传入的OpenAI格式请求，并将其转换为其他上游提供商（如Gemini）的格式。
同时，也负责将上游提供商的响应转换回OpenAI格式。
"""
import json
from typing import Dict, Any, Tuple, Optional, List
from app.services.preset_proxy.utils import _merge_messages

def to_gemini_request(openai_request: Dict[str, Any], preset_messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], str]:
    """
    将OpenAI格式的请求转换为Gemini格式。
    返回转换后的请求体和原始模型名称。
    """
    model = openai_request.get("model", "gemini-1.5-pro-latest")
    # 简单的模型映射，可以根据需要扩展
    if "gpt-4" in model:
        gemini_model = "gemini-1.5-pro-latest"
    elif "gpt-3.5" in model:
        gemini_model = "gemini-1.5-flash-latest"
    else:
        gemini_model = model # 假设用户可能直接指定了gemini模型

    # 合并预设消息和用户消息
    raw_messages = (preset_messages or []) + openai_request.get("messages", [])
    all_messages = _merge_messages(raw_messages)

    # 构建Gemini的contents
    contents = []
    system_prompt = None
    for message in all_messages:
        role = message.get("role")
        if role == "system":
            # Gemini通过system_instruction字段处理系统提示
            system_prompt = {"parts": [{"text": message.get("content")}]}
            continue
        
        # 转换角色
        gemini_role = "user" if role == "user" else "model"
        
        # 处理content字段（可能是字符串或数组）
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    # Gemini需要MIME类型和base64数据
                    if image_url.startswith("data:"):
                        parts.append(parse_data_url(image_url))

        contents.append({
            "role": gemini_role,
            "parts": parts
        })

    # 构建Gemini的generationConfig
    generation_config = {
        "temperature": openai_request.get("temperature", 0.7),
        "topP": openai_request.get("top_p", 1.0),
        "maxOutputTokens": openai_request.get("max_tokens", 2048),
    }
    if "stop" in openai_request and openai_request["stop"]:
        generation_config["stopSequences"] = [openai_request["stop"]] if isinstance(openai_request["stop"], str) else openai_request["stop"]

    gemini_body = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_prompt:
        gemini_body["system_instruction"] = system_prompt

    # 转换工具
    if "tools" in openai_request:
        gemini_body["tools"] = [{"functionDeclarations": [tool["function"] for tool in openai_request["tools"]]}]

    return gemini_body, gemini_model

def from_gemini_response(gemini_response: Dict[str, Any], original_model: str, is_stream: bool) -> Dict[str, Any]:
    """
    将Gemini的（非流式）响应转换为OpenAI格式。
    """
    choices = []
    for i, candidate in enumerate(gemini_response.get("candidates", [])):
        finish_reason = candidate.get("finishReason", "stop").lower()
        if finish_reason == "max_tokens":
            finish_reason = "length"

        message = {"role": "assistant", "content": None}
        parts = candidate.get("content", {}).get("parts", [])
        
        # 检查是否有工具调用
        tool_calls = []
        text_content = []
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{i}_{len(tool_calls)}", # 创建唯一ID
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": json.dumps(fc.get("args", {}))
                    }
                })
            elif "text" in part:
                text_content.append(part["text"])

        if tool_calls:
            message["tool_calls"] = tool_calls
        if text_content:
            message["content"] = "".join(text_content)

        choices.append({
            "index": i,
            "message": message,
            "finish_reason": finish_reason,
        })
    
    usage = gemini_response.get("usageMetadata", {})
    return {
        "id": f"chatcmpl-{hash(json.dumps(gemini_response))}",
        "object": "chat.completion",
        "created": int(json.dumps(gemini_response.get("createTime", 0))),
        "model": original_model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }

def from_gemini_stream_chunk(gemini_chunk: Dict[str, Any], original_model: str) -> Optional[Dict[str, Any]]:
    """
    将Gemini的流式数据块转换为OpenAI格式的流式数据块。
    """
    if not gemini_chunk.get("candidates"):
        return None

    delta = {}
    candidate = gemini_chunk["candidates"][0]
    
    # 提取文本或工具调用
    content_part = candidate.get("content", {}).get("parts", [{}])[0]
    if "text" in content_part:
        delta = {"role": "assistant", "content": content_part["text"]}
    elif "functionCall" in content_part:
         fc = content_part["functionCall"]
         delta = {
             "tool_calls": [{
                 "index": 0,
                 "id": f"call_{hash(json.dumps(fc))}",
                 "type": "function",
                 "function": {
                     "name": fc.get("name"),
                     "arguments": json.dumps(fc.get("args", ""))
                 }
             }]
         }

    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason.lower() != "unspecified":
        finish_reason = finish_reason.lower()
        if finish_reason == "max_tokens":
            finish_reason = "length"

    choice = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason
    }

    return {
        "id": f"chatcmpl-stream-{hash(json.dumps(gemini_chunk))}",
        "object": "chat.completion.chunk",
        "created": int(json.dumps(gemini_chunk.get("createTime", 0))),
        "model": original_model,
        "choices": [choice],
    }

def parse_data_url(data_url: str) -> Dict[str, Any]:
    """
    解析Data URL，提取MIME类型和base64数据。
    """
    try:
        header, encoded = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        return {
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }
    except Exception:
        return {}

def from_gemini_to_openai_models(gemini_models_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    将Gemini格式的模型列表转换为OpenAI格式。
    """
    openai_models = []
    for model in gemini_models_response.get("models", []):
        model_id = model.get("name", "").split('/')[-1]
        openai_models.append({
            "id": model_id,
            "object": "model",
            "created": 0, # OpenAI API需要此字段，但Gemini不提供，设为0
            "owned_by": "google"
        })
    
    return {
        "object": "list",
        "data": openai_models
    }