from typing import Dict, Any
from app.schemas.openai import GeneralOpenAIRequest, Message

def to_general_openai_request(request_body: Dict[str, Any], incoming_format: str, is_stream: bool) -> GeneralOpenAIRequest:
    """
    Converts a request from any supported format to a GeneralOpenAIRequest.
    """
    if incoming_format == "openai":
        return GeneralOpenAIRequest(**request_body)
    
    if incoming_format == "gemini":
        messages = []
        contents = request_body.get("contents", [])
        for content in contents:
            role = content.get("role", "user")
            if role == "model":
                role = "assistant"
            
            text_content = "".join([p.get("text", "") for p in content.get("parts", [])])
            messages.append(Message(role=role, content=text_content))
        
        gen_config = request_body.get("generationConfig", {})
        
        return GeneralOpenAIRequest(
            model=request_body.get("model", "unknown"),
            messages=messages,
            stream=is_stream,
            temperature=gen_config.get("temperature"),
            top_p=gen_config.get("topP"),
            max_tokens=gen_config.get("maxOutputTokens")
        )
        
    raise ValueError(f"Unsupported incoming format: {incoming_format}")