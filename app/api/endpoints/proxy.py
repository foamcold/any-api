import json
import time
import httpx
import logging
from typing import Any, List, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api import deps
from app.models.user import User
from app.models.key import ExclusiveKey, OfficialKey
from app.models.preset import Preset
from app.models.regex import RegexRule
from app.models.preset_regex import PresetRegexRule
from app.models.log import Log
from app.models.system_config import SystemConfig
from app.schemas.openai import ChatCompletionRequest
from app.services.chat_processor import chat_processor
from app.core.config import settings

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    此端点现在专门用于处理预设密钥 (gapi-) 的请求。
    所有官方密钥（非 gapi-）的请求都应由根目录下的 /v1 和 /v1beta 路由处理。
    """
    # 1. Auth & Key Validation
    official_key, user = key_info
    
    # get_official_key_from_proxy 保证了只有 gapi- 密钥会到达这里
    # 并且 user 对象一定存在。
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid key. This endpoint is for preset keys (gapi-) only."
        )

    # 为了日志记录和 ChatProcessor，我们需要获取 exclusive_key 对象
    auth_header = request.headers.get("Authorization")
    client_key = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else ""
    if not client_key:
        raise HTTPException(status_code=401, detail="Authorization header is missing or invalid.")

    result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.key == client_key))
    exclusive_key = result.scalars().first()
    if not exclusive_key:
        # 这种情况理论上不应该发生，因为 get_official_key_from_proxy 已经验证过
        raise HTTPException(status_code=401, detail="Invalid preset key.")

    logging.debug(f"处理专属 Key 请求. Key ID: {exclusive_key.id}, 名称: {exclusive_key.name}")

    # 2. 调用 ChatProcessor
    # ChatProcessor 内部处理请求解析、日志记录等所有逻辑
    result = await chat_processor.process_request(
        request=request,
        db=db,
        official_key=official_key,
        exclusive_key=exclusive_key,
        user=user,
        background_tasks=background_tasks,
        original_format="openai"
    )
    
    if isinstance(result, AsyncGenerator):
        # 流式响应
        return StreamingResponse(result, media_type="text/event-stream")
    else:
        # 非流式响应
        response_content, status_code, _ = result
        return JSONResponse(content=response_content, status_code=status_code)
