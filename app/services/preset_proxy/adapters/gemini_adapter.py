import json
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

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

def from_openai_stream_chunk(openai_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    将OpenAI的流式数据块转换为Gemini格式的流式数据块。
    """
    logger.debug(f"开始转换OpenAI流式块: {openai_chunk}")
    if not openai_chunk.get("choices"):
        logger.debug("块中无 'choices'，返回 None")
        return None

    choice = openai_chunk["choices"][0]
    delta = choice.get("delta", {})
    parts = []

    # 1. 处理内容 (content)
    if "content" in delta and delta["content"]:
        logger.debug(f"提取到内容: {delta['content']}")
        parts.append({"text": delta["content"]})
    
    # 2. 处理工具调用 (tool_calls)
    if "tool_calls" in delta:
        for tool_call in delta["tool_calls"]:
            func = tool_call.get("function", {})
            logger.debug(f"提取到工具调用: {func}")
            parts.append({
                "functionCall": {
                    "name": func.get("name"),
                    "args": json.loads(func.get("arguments", "{}"))
                }
            })

    # 3. 处理结束原因 (finish_reason)
    finish_reason = None
    raw_reason = choice.get("finish_reason")
    if raw_reason:
        logger.debug(f"提取到结束原因: {raw_reason}")
        if raw_reason == "stop": finish_reason = "STOP"
        elif raw_reason == "length": finish_reason = "MAX_TOKENS"
        elif raw_reason == "content_filter": finish_reason = "SAFETY"
        else: finish_reason = "FINISH_REASON_UNSPECIFIED"

    # 核心逻辑：如果既没有内容部分，也没有结束原因，则这是一个空的更新，应跳过
    if not parts and not finish_reason:
        logger.debug("块中无有效内容或结束原因，返回 None")
        return None

    # 构建 candidate
    candidate = { "index": choice.get("index", 0) }
    if parts:
        candidate["content"] = {"parts": parts, "role": "model"}
    if finish_reason:
        candidate["finishReason"] = finish_reason

    final_gemini_chunk = {"candidates": [candidate]}
    logger.debug(f"成功转换为Gemini流式块: {final_gemini_chunk}")
    return final_gemini_chunk