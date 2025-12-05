import logging
import json
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.services.llm_proxy.main import LLMProxyService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_key_info),
    system_config: dict = Depends(deps.get_system_config)
):
    """
    统一处理 /v1/chat/completions 请求，并根据密钥类型分派到不同的服务。
    """
    client_key, key_obj, user = key_info
    
    logger.debug(f"路由 /v1/chat/completions：准备根据密钥类型分派。客户端密钥: '{client_key}'")

    # 检查是否为伪流请求
    body = await request.json()
    model_name = body.get("model", "")
    is_pseudo_stream = False
    if system_config and system_config.pseudo_streaming_enabled and model_name.startswith("伪流/"):
        is_pseudo_stream = True
        body["model"] = model_name.replace("伪流/", "", 1)

    is_stream_override = False if is_pseudo_stream else None

    if client_key.startswith("gapi-"):
        logger.debug("检测到 gapi- 密钥，正在分派到 PresetProxyService...")
        from app.services.preset_proxy.main import PresetProxyService
        service = PresetProxyService(
            db=db,
            exclusive_key_obj=key_obj,
            user=user,
            incoming_format="openai",
            is_stream_override=is_stream_override
        )
        return await service.proxy_request(body, is_pseudo_stream)
    else:
        logger.debug("检测到非 gapi- 密钥，正在分派到 LLMProxyService...")
        if client_key.startswith("sk-"):
            target_provider = "openai"
        elif client_key.startswith("AIza"):
            target_provider = "gemini"
        else:
            target_provider = "openai"

        proxy_service = LLMProxyService(
            db=db,
            official_key_obj=key_obj,
            user=user,
            incoming_format="openai", # /v1 接收OpenAI格式
            target_provider=target_provider,
            is_stream_override=is_stream_override
        )
        return await proxy_service.proxy_request(body, is_pseudo_stream)

@router.get("/models")
async def list_models(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_key_info),
    system_config: dict = Depends(deps.get_system_config)
):
    """
    统一处理 /v1/models 请求，并根据密钥类型分派。
    """
    client_key, key_obj, user = key_info
    
    logger.debug(f"路由 /v1/models：准备根据密钥类型分派。客户端密钥: '{client_key}'")

    if client_key.startswith("gapi-"):
        logger.debug("检测到 gapi- 密钥，正在分派到 PresetProxyService...")
        from app.services.preset_proxy.main import PresetProxyService
        service = PresetProxyService(
            db=db,
            exclusive_key_obj=key_obj,
            user=user,
            incoming_format="openai"
        )
        return await service.proxy_list_models(request, system_config)
    else:
        logger.debug("检测到非 gapi- 密钥，正在分派到 LLMProxyService...")
        if client_key.startswith("sk-"):
            target_provider = "openai"
        elif client_key.startswith("AIza"):
            target_provider = "gemini"
        else:
            target_provider = "openai"

        proxy_service = LLMProxyService(
            db=db,
            official_key_obj=key_obj,
            user=user,
            incoming_format="openai",
            target_provider=target_provider
        )
        return await proxy_service.proxy_list_models(request, system_config)