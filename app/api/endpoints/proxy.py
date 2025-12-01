import json
import time
import httpx
import logging
from typing import Any, List, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request, Response
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
from app.services.gemini_service import gemini_service
from app.services.universal_converter import universal_converter
from app.services.variable_service import variable_service
from app.services.regex_service import regex_service
from app.services.chat_processor import chat_processor
from app.services.proxy_service import proxy_service
from app.core.config import settings

router = APIRouter()

# Configure logger
logger = logging.getLogger(__name__)
current_log_level = "INFO"

async def get_log_level(db: AsyncSession):
    global current_log_level
    result = await db.execute(select(SystemConfig))
    config = result.scalars().first()
    if config and config.log_level:
        current_log_level = config.log_level
        return config.log_level
    current_log_level = "INFO"
    return "INFO"

def update_logger_level(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Ensure handler exists and set level for handler as well
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    for handler in logger.handlers:
        handler.setLevel(level)

def debug_log(message: str):
    """
    Wrapper for debug logging.
    """
    if current_log_level == "DEBUG":
        logger.debug(message)


@router.get("/v1/models")
async def list_models(
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    处理 GET /v1/models 请求，通过代理到 Google API 列出可用模型。
    使用新的依赖项处理密钥。
    """
    official_key, _ = key_info

    # 2. 代理到 Google API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": official_key.key}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"请求 Google API 时出错: {e}")

    # 3. 转换响应
    try:
        gemini_response = response.json()
        models = gemini_response.get("models", [])
        
        openai_models = []
        for model in models:
            model_id = model.get("name", "").replace("models/", "")
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            })
            
        return {
            "object": "list",
            "data": openai_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析或转换模型列表时出错: {e}")


@router.api_route("/v1beta/{path:path}", methods=["POST", "PUT", "DELETE", "GET"])
async def proxy_beta_requests(
    request: Request,
    path: str,
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    通用代理，处理 /v1beta/ 下的所有请求，并以流的形式返回响应。
    修复了错误状态码无法正确透传的问题。
    """
    official_key, _ = key_info
    
    target_url = f"https://generativelanguage.googleapis.com/v1beta/{path}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "authorization", "x-goog-api-key", "key"]}
    params = dict(request.query_params)
    params['key'] = official_key.key
    body = await request.body()

    try:
        client = httpx.AsyncClient(timeout=120.0)
        
        req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=params,
            content=body
        )
        
        response = await client.send(req, stream=True)
        
        if response.status_code >= 400:
            error_content = await response.aread()
            await response.aclose()
            await client.aclose()
            return Response(content=error_content, status_code=response.status_code, media_type=response.headers.get("content-type"))
            
        excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        response_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_headers}

        async def safe_stream_generator(response):
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except (httpx.ReadError, httpx.ConnectError) as e:
                logger.error(f"Proxy stream connection error: {e}")
            except Exception as e:
                logger.error(f"Unexpected proxy stream error: {e}")
            finally:
                await response.aclose()

        return StreamingResponse(
            safe_stream_generator(response),
            status_code=response.status_code,
            headers=response_headers,
            background=None # background task is handled in finally block
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"连接 Google API 失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    # 0. Configure Logging Level
    log_level = await get_log_level(db)
    update_logger_level(log_level)

    # 1. Auth & Key Validation
    official_key, user = key_info
    
    # 检查是否是专属密钥的逻辑现在由 get_official_key_from_proxy 处理
    # 如果 user 不为 None, 则说明是有效的专属密钥
    is_exclusive = user is not None
    exclusive_key = None
    if is_exclusive:
        # 为了日志记录，可能需要获取 exclusive_key 对象
        auth_header = request.headers.get("Authorization")
        client_key = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else ""
        if client_key:
            result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.key == client_key))
            exclusive_key = result.scalars().first()
            debug_log(f"处理专属 Key 请求. Key ID: {exclusive_key.id}, 名称: {exclusive_key.name}")
    else:
        debug_log(f"处理官方 Key 请求.")

    # 2. Parse Request
    try:
        body = await request.json()
        openai_request = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # 3. 如果是 gapi- key, 调用 ChatProcessor
    if is_exclusive and exclusive_key:
        start_time = time.time()
        
        # We need to handle the streaming response differently to log it.
        # Let's create a wrapper generator.
        async def logging_streaming_response_generator(generator: AsyncGenerator) -> AsyncGenerator[bytes, None]:
            nonlocal ttft
            first_chunk_received = False
            full_response_content = ""
            try:
                async for chunk in generator:
                    if not first_chunk_received:
                        ttft = time.time() - start_time
                        first_chunk_received = True
                    
                    # Assuming chunk is "data: {...}\n\n"
                    if chunk.startswith(b'data: '):
                        content_part = chunk[6:].strip()
                        if content_part != b'[DONE]':
                            try:
                                json_content = json.loads(content_part)
                                if json_content.get('choices'):
                                    delta = json_content['choices'][0].get('delta', {})
                                    full_response_content += delta.get('content', '')
                            except json.JSONDecodeError:
                                pass # Ignore json parsing errors for now
                    yield chunk
            finally:
                latency = time.time() - start_time
                # Assuming simple token calculation for now
                input_tokens = len(str(body)) // 4
                output_tokens = len(full_response_content) // 4
                
                log_entry = Log(
                    exclusive_key_id=exclusive_key.id,
                    user_id=user.id,
                    model=openai_request.model,
                    status="ok",
                    status_code=200,
                    latency=latency,
                    ttft=ttft,
                    is_stream=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                db.add(log_entry)
                await db.commit()

        result = await chat_processor.process_request(
            request=request,
            db=db,
            official_key=official_key.key,
            exclusive_key=exclusive_key,
            user=user,
            log_level=log_level
        )
        
        ttft = 0.0
        # 根据结果类型返回响应
        if isinstance(result, AsyncGenerator):
            return StreamingResponse(logging_streaming_response_generator(result), media_type="text/event-stream")
        else:
            response_content, status_code, _ = result
            latency = time.time() - start_time
            
            input_tokens = response_content.get('usage', {}).get('prompt_tokens', 0)
            output_tokens = response_content.get('usage', {}).get('completion_tokens', 0)
            
            status = "ok" if status_code == 200 else "error"
            
            log_entry = Log(
                exclusive_key_id=exclusive_key.id,
                user_id=user.id,
                model=openai_request.model,
                status=status,
                status_code=status_code,
                latency=latency,
                ttft=latency, # For non-streaming, ttft is same as latency
                is_stream=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            db.add(log_entry)
            await db.commit()
            
            return JSONResponse(content=response_content, status_code=status_code)

    # --- 非 gapi- key 的新逻辑 (使用 ProxyService) ---
    
    return await proxy_service.smart_proxy_handler(
        request=request,
        db=db,
        path="chat/completions",
        official_key_obj=official_key,
        user=user, # user will be None for non-exclusive keys
        incoming_format="openai"
    )
