from typing import Generator, Optional, Tuple
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
from app.services.claude_service import claude_service
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
            print("错误：Turnstile已启用但未配置Secret Key")
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

async def get_official_key_from_proxy(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Tuple[OfficialKey, Optional[User]]:
    """
    从代理请求中提取、验证并返回一个有效的官方API密钥。
    - 如果提供的是专属密钥 (gapi-...), 则验证并返回一个轮询的官方密钥。
    - 如果提供的是普通密钥, 则直接返回。
    - 同时返回关联的用户对象（如果存在）。
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        # 兼容某些客户端可能使用的 x-goog-api-key 或 key 参数
        client_key = request.headers.get("x-goog-api-key") or request.query_params.get("key")
        if not client_key:
            print("DEBUG: deps - 未找到 API 密钥")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供 API 密钥")
    else:
        client_key = auth_header.split(" ")[1]

    print(f"DEBUG: deps - 提取到的 Key: {client_key}, 来源: {'Auth Header' if auth_header else 'Query/X-Header'}")

    if client_key and client_key.startswith("gapi-"):
        # 是专属密钥，需要验证并轮询
        result = await db.execute(
            select(ExclusiveKey).filter(ExclusiveKey.key == client_key, ExclusiveKey.is_active == True)
        )
        exclusive_key = result.scalars().first()
        
        if not exclusive_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的专属密钥")
            
        user_result = await db.execute(select(User).filter(User.id == exclusive_key.user_id))
        user = user_result.scalars().first()
        
        # 必须绑定渠道
        if not exclusive_key.channel_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该密钥未绑定任何渠道")

        channel_res = await db.execute(select(Channel).filter(Channel.id == exclusive_key.channel_id))
        channel = channel_res.scalars().first()
        
        if not channel:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="绑定的渠道不存在")

        channel_type = channel.type.lower()
        official_key = None
        
        print(f"DEBUG: deps - 专属密钥绑定渠道: ID={channel.id}, 名称={channel.name}, 类型={channel_type}")
        
        official_key_obj = None
        try:
            if channel_type == "claude":
                official_key_obj = await claude_service.get_active_key(db, channel_id=channel.id)
            elif channel_type in ["openai", "gemini"]:
                official_key_obj = await gemini_service.get_active_key(db, channel_id=channel.id)
            else:  # 为其他可能的类型提供一个默认回退
                official_key_obj = await gemini_service.get_active_key(db, channel_id=channel.id)

            if not official_key_obj:
                error_detail = f"渠道 '{channel.name}' (类型: {channel_type}, ID: {channel.id}) 下没有可用的官方密钥。"
                raise HTTPException(status_code=503, detail=error_detail)

            print(f"DEBUG: deps - 成功从渠道 {channel.name} 获取到官方密钥: {official_key_obj.key[:8]}...")
            return official_key_obj, user

        except HTTPException as e:
            # 重复抛出已知的HTTP异常
            raise e
        except Exception as e:
            # 捕获在获取密钥期间可能发生的任何其他意外错误
            raise HTTPException(status_code=500, detail=f"获取密钥时发生未知错误: {e}")
    else:
        # 是普通密钥, 无法直接关联到数据库中的对象，日志记录功能将受限
        # 为了兼容性，我们创建一个临时的、不保存到数据库的 OfficialKey 对象
        temp_key_obj = OfficialKey(key=client_key)
        return temp_key_obj, None
