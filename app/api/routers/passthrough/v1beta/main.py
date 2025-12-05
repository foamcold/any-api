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
    key_info: tuple = Depends(deps.get_key_info),
    system_config: dict = Depends(deps.get_system_config)
):
    """
    统一处理 /v1beta/models/{model}:{action} 请求，并根据密钥类型分派。
    """
    client_key, key_obj, user = key_info

    logger.debug(f"路由 /v1beta/models：准备根据密钥类型分派。客户端密钥: '{client_key}'")

    body = await request.json()
    is_client_stream = ":streamGenerateContent" in model_and_action

    # 检查是否为伪流请求
    model_name = model_and_action.split(":")[0]
    is_pseudo_stream = False
    if system_config and system_config.pseudo_streaming_enabled and model_name.startswith("伪流/"):
        is_pseudo_stream = True
        true_model_name = model_name.replace("伪流/", "", 1)
        model_and_action = f"{true_model_name}:{model_and_action.split(':')[1]}"

    # 确定对上游的流式覆盖。伪流模式下，上游必须是非流式。
    is_stream_override = False if is_pseudo_stream else is_client_stream

    # 更新 body
    body['model'] = model_and_action.split(":")[0]
    body['stream'] = is_client_stream

    if client_key.startswith("gapi-"):
        logger.debug("检测到 gapi- 密钥，正在分派到 PresetProxyService...")
        from app.services.preset_proxy.main import PresetProxyService
        
        service = PresetProxyService(
            db=db,
            exclusive_key_obj=key_obj,
            user=user,
            incoming_format="gemini",
            is_stream_override=is_stream_override
        )
        return await service.proxy_request(body, is_pseudo_stream)
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
        return await service.proxy_list_models(request, system_config)
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
        return await proxy_service.proxy_list_models(request, system_config)
