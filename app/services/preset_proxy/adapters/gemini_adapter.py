from typing import Dict, Any, Tuple, List

def to_openai_request(gemini_request: Dict[str, Any], preset_messages: List[Dict[str, str]], is_stream: bool) -> Tuple[Dict[str, Any], str]:
    """
    将 Gemini 请求格式转换为 OpenAI 请求格式，并注入预设。
    """
    
    # 1. 转换 Gemini 的 contents
    converted_messages = []
    contents = gemini_request.get("contents", [])
    for content in contents:
        role = content.get("role")
        if role == "model":
            role = "assistant"
        
        # 确保角色对于OpenAI是有效的
        if role not in ["user", "assistant", "system"]:
            # 跳过无效角色或进行适当处理
            continue

        text_content = "".join([p.get("text", "") for p in content.get("parts", [])])
        converted_messages.append({"role": role, "content": text_content})

    # 2. 合并预设和转换后的消息，处理连续的同角色消息
    all_messages = preset_messages + converted_messages
    if not all_messages:
        final_messages = []
    else:
        final_messages = [all_messages[0]]
        for i in range(1, len(all_messages)):
            # 如果当前消息的角色与最终消息列表中的最后一个消息角色相同
            if all_messages[i]["role"] == final_messages[-1]["role"]:
                # 合并内容
                final_messages[-1]["content"] += "\n" + all_messages[i]["content"]
            else:
                final_messages.append(all_messages[i])

    # 从原始请求中获取模型名称
    model_name = gemini_request.get("model", "gemini-pro")
    
    payload = {
        "model": model_name,
        "messages": final_messages,
        "stream": is_stream
    }

    # 映射 generationConfig
    gen_config = gemini_request.get("generationConfig", {})
    if "temperature" in gen_config:
        payload["temperature"] = gen_config["temperature"]
    if "topP" in gen_config:
        payload["top_p"] = gen_config["topP"]
    if "maxOutputTokens" in gen_config:
        payload["max_tokens"] = gen_config["maxOutputTokens"]
    
    return payload, model_name

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