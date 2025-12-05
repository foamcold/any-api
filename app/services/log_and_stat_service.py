import logging
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Union

from app.models.log import Log
from app.models.key import OfficialKey, ExclusiveKey
from app.models.user import User

logger = logging.getLogger(__name__)

async def update_key_and_log_usage(
    db: AsyncSession,
    key_obj: Union[OfficialKey, ExclusiveKey],
    user: Optional[User],
    model: str,
    input_tokens: int,
    output_tokens: int,
    status_code: int,
    latency: float,
    ttft: float = 0.0,
    is_stream: bool = False,
    status: str = "ok"
):
    """
    统一处理日志记录和密钥统计更新。
    """
    try:
        # 1. 创建日志记录
        log_entry = Log(
            user_id=user.id if user else None,
            model=model,
            status=status,
            status_code=status_code,
            latency=latency,
            ttft=ttft,
            is_stream=is_stream,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # 根据密钥类型关联日志
        if isinstance(key_obj, OfficialKey):
            log_entry.official_key_id = key_obj.id
        elif isinstance(key_obj, ExclusiveKey):
            log_entry.exclusive_key_id = key_obj.id

        db.add(log_entry)

        # 2. 更新密钥统计信息 (仅针对 OfficialKey)
        if isinstance(key_obj, OfficialKey):
            key_obj.usage_count += 1
            key_obj.input_tokens += input_tokens
            key_obj.output_tokens += output_tokens
            key_obj.last_status_code = status_code
            
            if status == "error":
                key_obj.error_count += 1
                key_obj.last_status = "error"
            else:
                key_obj.last_status = "active"
            
            db.add(key_obj)

        # 3. 提交事务
        await db.commit()
        logger.debug(f"成功记录日志并更新密钥统计。模型: {model}, 输入Token: {input_tokens}, 输出Token: {output_tokens}")

    except Exception as e:
        await db.rollback()
        logger.error(f"记录日志和更新密钥统计时发生错误: {e}", exc_info=True)
