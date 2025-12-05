import json
import logging
from fastapi import APIRouter, Depends, Request, Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.services.llm_proxy.main import LLMProxyService

router = APIRouter()
logger = logging.getLogger(__name__)

# Gemini API 路径通常包含模型名称，例如 /v1beta/models/gemini-1.5-pro:generateContent
# 我们使用 path parameter 来捕获完整的模型和动作
@router.post("/models/{model_and_action:path}")
async def generate_content(
    request: Request,
    model_and_action: str = Path(..., description="模型名称和动作，例如 'gemini-1.5-pro:generateContent'"),
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    处理 /v1beta/models/{model}:{action} 请求。
    根据密钥类型（AIza 或 sk-）决定目标上游。
    """
    official_key, user = key_info
    client_key = official_key.key

    if client_key.startswith("AIza"):
        target_provider = "gemini"
    elif client_key.startswith("sk-"):
        target_provider = "openai"
    else:
        # 默认或可以返回错误
        target_provider = "gemini"
        logger.warning(f"无法从Key前缀确定目标服务商，将默认使用: {target_provider}")

    # 检查是否为流式请求
    is_stream = "streamGenerateContent" in model_and_action

    # 从路径中提取模型名称并附加到请求体中
    body = await request.json()
    model_name = model_and_action.split(":")[0]
    body['model'] = model_name

    # 创建一个新的Request对象，因为原始的request.body()只能读取一次
    # 我们需要一个有状态的 receive 函数，因为它会被多次调用。
    _body_bytes = json.dumps(body).encode('utf-8')
    _stream_sent = False
    async def receive():
        nonlocal _stream_sent
        if not _stream_sent:
            _stream_sent = True
            return {'type': 'http.request', 'body': _body_bytes, 'more_body': False}
        return {'type': 'http.disconnect'}

    new_request = Request(scope=request.scope, receive=receive)

    proxy_service = LLMProxyService(
        db=db,
        official_key_obj=official_key,
        user=user,
        incoming_format="gemini", # /v1beta 接收Gemini格式
        target_provider=target_provider,
        is_stream_override=is_stream
    )
    
    return await proxy_service.proxy_request(new_request)
