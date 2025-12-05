import logging
from typing import Generator, Optional, Tuple, Union
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core import security
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.models.key import ExclusiveKey, OfficialKey
from app.schemas.token import TokenPayload
from app.services.gemini_service import gemini_service
from app.models.channel import Channel
from app.models.system_config import SystemConfig
from app.services.turnstile_service import turnstile_service
from fastapi import Body
from app.services.captcha_service import captcha_service
from app.models.verification_code import VerificationCode


async def verify_captcha(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    依赖项：验证图形验证码
    - 兼容从 JSON body 和 cookie 中获取 captcha_id
    """
    config_result = await db.execute(select(SystemConfig).filter(SystemConfig.id == 1))
    system_config = config_result.scalars().first()

    if system_config and system_config.enable_captcha:
        captcha_id: Optional[str] = None
        captcha_code: Optional[str] = None

        # 1. 尝试从 JSON body 获取
        try:
            body = await request.json()
            captcha_id = body.get('captcha_id')
            captcha_code = body.get('captcha_code')
        except Exception:
            pass # 如果不是json请求则忽略

        # 2. 如果在 body 中找不到 captcha_id，则尝试从 cookie 获取
        if not captcha_id:
            captcha_id = request.cookies.get("captcha_id")
        
        # 3. 如果在 body 中找不到 captcha_code, 尝试从 form 获取 (兼容不同请求类型)
        if not captcha_code:
            try:
                form = await request.form()
                captcha_code = form.get('captcha_code')
            except Exception:
                pass

        if not captcha_id or not captcha_code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="需要图形验证码",
            )
        
        is_valid = await captcha_service.verify_captcha(db, captcha_id, captcha_code)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="验证码错误",
            )


reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.VITE_API_STR}/auth/login/access-token"
)

async def verify_turnstile(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    依赖项：验证Cloudflare Turnstile token
    """
    config_result = await db.execute(select(SystemConfig).filter(SystemConfig.id == 1))
    system_config = config_result.scalars().first()

    if system_config and system_config.enable_turnstile:
        if not system_config.turnstile_secret_key:
            # 如果启用了但未配置密钥，则跳过验证并记录错误
            logging.error("Turnstile已启用但未配置Secret Key")
            return

        # 尝试从 JSON body 或 form data 中获取 token
        turnstile_token: Optional[str] = None
        try:
            body = await request.json()
            turnstile_token = body.get('turnstile_token')
        except Exception:
            try:
                form = await request.form()
                turnstile_token = form.get('turnstile_token')
            except Exception:
                pass

        if not turnstile_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="需要人机验证",
            )
        
        client_ip = request.client.host if request.client else None
        
        turnstile_service.configure(system_config.turnstile_secret_key)
        is_valid = await turnstile_service.verify_token(turnstile_token, ip=client_ip)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="人机验证失败",
            )

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(reusable_oauth2)
) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无法验证凭据",
        )
    
    result = await db.execute(select(User).filter(User.id == int(token_data.sub)))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    return current_user

async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role != "super_admin":
        raise HTTPException(
            status_code=403, detail="需要超级管理员权限"
        )
    return current_user

async def get_current_active_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=403, detail="需要管理员权限"
        )
    return current_user

async def get_optional_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    token = request.headers.get("Authorization")
    if token:
        try:
            # 去除 "Bearer " 前缀
            token = token.split(" ")[1]
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
            )
            token_data = TokenPayload(**payload)
            result = await db.execute(select(User).filter(User.id == int(token_data.sub)))
            user = result.scalars().first()
            return user
        except (JWTError, ValidationError, IndexError):
            # Token 无效或格式错误
            return None
    return None

async def get_key_info(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Tuple[str, Union[ExclusiveKey, OfficialKey], Optional[User]]:
    """
    统一的密钥处理依赖项。
    返回 (客户端密钥, 密钥对象, 用户对象)。
    """
    client_key = _get_client_key(request)
    if not client_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供 API 密钥。")

    if client_key.startswith("gapi-"):
        from sqlalchemy.orm import selectinload
        stmt = (
            select(ExclusiveKey)
            .options(selectinload(ExclusiveKey.preset), selectinload(ExclusiveKey.channel))
            .filter(ExclusiveKey.key == client_key, ExclusiveKey.is_active == True)
        )
        result = await db.execute(stmt)
        exclusive_key = result.scalars().first()

        if not exclusive_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或已禁用的预设密钥。")

        user_result = await db.execute(select(User).filter(User.id == exclusive_key.user_id))
        user = user_result.scalars().first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="密钥关联的用户不存在。")

        return client_key, exclusive_key, user
    else:
        # 对于非 gapi- 密钥，我们假设它是一个官方密钥，并创建一个临时对象。
        temp_key_obj = OfficialKey(key=client_key)
        return client_key, temp_key_obj, None

def _get_client_key(request: Request) -> Optional[str]:
    """从请求中提取客户端 API 密钥的辅助函数。"""
    logging.debug(f"检查请求中的密钥, Path: {request.url.path}, Headers: {request.headers}")
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        key = auth_header.split(" ")[1]
        logging.debug(f"从 Authorization Header 中找到密钥: {key[:10]}...")
        return key
    
    # 兼容其他常见的位置
    key = request.headers.get("x-api-key") or request.query_params.get("key") or request.headers.get("x-goog-api-key")
    if key:
        logging.debug(f"从 x-api-key 或查询参数中找到密钥: {key[:10]}...")
    else:
        logging.debug("在任何位置都未找到密钥。")
    return key

async def get_system_config(db: AsyncSession = Depends(get_db)) -> Optional[SystemConfig]:
   """
   依赖项：获取系统配置。
   """
   result = await db.execute(select(SystemConfig).limit(1))
   config = result.scalars().first()
   if not config:
       # 在没有找到配置时可以返回一个默认配置对象或None
       # 这里返回None，让调用者处理
       logging.warning("未在数据库中找到系统配置。")
       return None
   return config
