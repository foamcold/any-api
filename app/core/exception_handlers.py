"""
全局异常处理器

替代中间件方案，使用 FastAPI 的 exception_handler 机制来捕获和格式化错误。
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.core.errors import create_api_error, ApiFormat

logger = logging.getLogger(__name__)

def _detect_client_format(request: Request) -> ApiFormat:
    """
    根据请求特征判断客户端期望的 API 格式
    
    判断依据（优先级从高到低）：
    1. 请求路径
    2. Header 特征
    3. Content-Type
    
    返回: "openai" | "gemini"
    """
    path = request.url.path
    
    # === 路径判断（最可靠）===
    
    # Gemini 特征路径: /v1beta, /gemini
    if path.startswith("/v1beta") or "/v1beta/" in path:
        return "gemini"
    
    if path.startswith("/gemini/") or path == "/gemini":
        return "gemini"
    
    # OpenAI 特征路径: /v1/chat/completions, /v1/models, /openai
    if path.startswith("/openai/") or path == "/openai":
        return "openai"
    
    if "/chat/completions" in path or "/completions" in path:
        return "openai"
    
    # === Header 判断 ===
    
    # Gemini 使用 x-goog-api-key
    if request.headers.get("x-goog-api-key"):
        return "gemini"
    
    # OpenAI 使用 Authorization: Bearer
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return "openai"
    
    # === 默认判断 ===
    
    # /v1 路径默认为 OpenAI
    if path.startswith("/v1"):
        return "openai"
    
    # 最终默认为 OpenAI 格式
    return "openai"

async def api_exception_handler(request: Request, exc: HTTPException):
    """
    处理所有 HTTPException
    """
    api_format = _detect_client_format(request)
    
    logger.info(f"捕获 HTTPException [{exc.status_code}] {request.url.path}: {exc.detail}, 格式: {api_format}")
    
    # 如果 detail 已经是字典格式（使用了 create_api_error），直接返回
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # 否则，创建对应格式的错误响应
    error_response = create_api_error(
        message=str(exc.detail),
        status_code=exc.status_code,
        api_format=api_format
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理参数验证错误 (422)
    """
    api_format = _detect_client_format(request)
    
    error_msg = f"参数验证错误: {exc.errors()}"
    # 简化错误信息
    try:
        # 尝试翻译常见的 Pydantic 错误信息
        raw_msg = exc.errors()[0]['msg']
        if raw_msg == "field required":
            msg_cn = "缺少必填字段"
        elif "value is not a valid" in raw_msg:
            msg_cn = "值无效"
        else:
            msg_cn = raw_msg
            
        error_msg = f"请求无效: {msg_cn} (位置: {exc.errors()[0]['loc']})"
    except:
        pass

    logger.info(f"捕获 Validation Error {request.url.path}, 格式: {api_format}")

    error_response = create_api_error(
        message=error_msg,
        status_code=422,
        api_format=api_format
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response
    )

async def general_exception_handler(request: Request, exc: Exception):
    """
    处理所有未捕获的异常 (500)
    """
    api_format = _detect_client_format(request)
    
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    
    error_response = create_api_error(
        message=f"服务器内部错误: {str(exc)}",
        status_code=500,
        api_format=api_format
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )
