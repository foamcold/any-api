from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.services.llm_proxy.main import LLMProxyService # 将在下一步创建

router = APIRouter()

@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(deps.get_db),
    key_info: tuple = Depends(deps.get_official_key_from_proxy)
):
    """
    处理 /v1/chat/completions 请求。
    根据密钥类型（sk- 或 AIza）决定目标上游。
    """
    official_key, user = key_info
    client_key = official_key.key

    if client_key.startswith("sk-"):
        target_provider = "openai"
    elif client_key.startswith("AIza"):
        target_provider = "gemini"
    else:
        # 默认或可以返回错误
        target_provider = "openai" 
        
    proxy_service = LLMProxyService(
        db=db,
        official_key_obj=official_key,
        user=user,
        incoming_format="openai", # /v1 接收OpenAI格式
        target_provider=target_provider
    )
    
    return await proxy_service.proxy_request(request)

# 可以在这里添加 /v1/models 等其他路由