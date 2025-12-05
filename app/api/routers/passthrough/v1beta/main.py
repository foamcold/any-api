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
    key_info: tuple = Depends(deps.get_key_info)
):
    """
    统一处理 /v1beta/models/{model}:{action} 请求，并根据密钥类型分派。
    """
    client_key, key_obj, user = key_info

    logger.debug(f"路由 /v1beta/models：准备根据密钥类型分派。客户端密钥: '{client_key}'")

    if client_key.startswith("gapi-"):
        logger.debug("检测到 gapi- 密钥，正在分派到 PresetProxyService...")
        from app.services.preset_proxy.main import PresetProxyService
        
        is_stream = ":streamGenerateContent" in model_and_action

        service = PresetProxyService(
            db=db,
            exclusive_key_obj=key_obj,
            user=user,
            incoming_format="gemini",
            is_stream_override=is_stream
        )
        
        # For preset mode, we need to reconstruct the request to pass the model name
        body = await request.json()
        body['model'] = model_and_action.split(":")[0]
        _body_bytes = json.dumps(body).encode('utf-8')
        _stream_sent = False
        async def receive():
            nonlocal _stream_sent
            if not _stream_sent:
                _stream_sent = True
                return {'type': 'http.request', 'body': _body_bytes, 'more_body': False}
            return {'type': 'http.disconnect'}
        new_request = Request(scope=request.scope, receive=receive)
        return await service.proxy_request(new_request)
    else:
        logger.debug("检测到非 gapi- 密钥，正在分派到 LLMProxyService...")
        if client_key.startswith("AIza"):
            target_provider = "gemini"
        elif client_key.startswith("sk-"):
            target_provider = "openai"
        else:
            target_provider = "gemini"

        is_stream = "streamGenerateContent" in model_and_action
        body = await request.json()
        body['model'] = model_and_action.split(":")[0]
        _body_bytes = json.dumps(body).encode('utf-8')
        _stream_sent = False
        async def receive():
            nonlocal _stream_sent
            if not _stream_sent:
                _stream_sent = True
                return {'type': 'http.request', 'body': _body_bytes, 'more_body': False}
            return {'type': 'http.disconnect'}
        
        # 重新构建请求时，必须包含原始的 headers
        new_scope = request.scope.copy()
        new_scope['headers'] = request.headers.raw
        new_request = Request(scope=new_scope, receive=receive)
        
        proxy_service = LLMProxyService(
            db=db,
            official_key_obj=key_obj,
            user=user,
            incoming_format="gemini",
            target_provider=target_provider,
            is_stream_override=is_stream
        )
        return await proxy_service.proxy_request(new_request)

@router.get("/models")
async def list_models(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_key_info)
):
    """
    统一处理 /v1beta/models 请求，并根据密钥类型分派。
    """
    client_key, key_obj, user = key_info
    
    logger.debug(f"路由 /v1beta/models：准备根据密钥类型分派。客户端密钥: '{client_key}'")

    if client_key.startswith("gapi-"):
        logger.debug("检测到 gapi- 密钥，正在分派到 PresetProxyService...")
        from app.services.preset_proxy.main import PresetProxyService
        service = PresetProxyService(
            db=db,
            exclusive_key_obj=key_obj,
            user=user,
            incoming_format="gemini"
        )
        return await service.proxy_list_models(request)
    else:
        logger.debug("检测到非 gapi- 密钥，正在分派到 LLMProxyService...")
        if client_key.startswith("AIza"):
            target_provider = "gemini"
        elif client_key.startswith("sk-"):
            target_provider = "openai"
        else:
            target_provider = "gemini"

        proxy_service = LLMProxyService(
            db=db,
            official_key_obj=key_obj,
            user=user,
            incoming_format="gemini",
            target_provider=target_provider
        )
        return await proxy_service.proxy_list_models(request)
