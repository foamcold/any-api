from typing import Dict, Any, Tuple, List

def to_openai_request(gemini_request: Dict[str, Any], preset_messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str]:
    """
    将 Gemini 请求格式转换为 OpenAI 请求格式，并注入预设。
    """
    messages = []
    
    # 注入预设
    for msg in preset_messages:
        messages.append(msg)
        
    # 转换 Gemini 的 contents
    contents = gemini_request.get("contents", [])
    for content in contents:
        role = content.get("role")
        if role == "model":
            role = "assistant"
        
        text_content = "".join([p.get("text", "") for p in content.get("parts", [])])
        messages.append({"role": role, "content": text_content})
        
    # 从原始请求中获取模型名称
    model_name = gemini_request.get("model", "gemini-pro") # 提供一个默认值以防万一
    
    return {
        "model": model_name,
        "messages": messages,
        # 其他参数的映射
    }, model_name

def from_openai_response(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 OpenAI 响应格式转换为 Gemini 响应格式。
    """
    candidates = []
    if "choices" in openai_response:
        for choice in openai_response["choices"]:
            message = choice.get("message", {})
            content = message.get("content")
            if content:
                candidates.append({
                    "content": {
                        "role": "model",
                        "parts": [{"text": content}]
                    },
                    "finishReason": "STOP" # 简化处理
                })
    return {"candidates": candidates}

def from_openai_stream_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 OpenAI 流式块转换为 Gemini 流式块。
    """
    choices = chunk.get("choices")
    if not choices:
        return {}

    delta = choices[0].get("delta", {})
    content = delta.get("content")
    
    if content:
        return {
            "candidates": [{
                "content": {
                    "parts": [{"text": content}]
                }
            }]
        }
    return {}