import httpx
import logging
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.models.key import OfficialKey
from app.models.system_config import SystemConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

# Basic Logger Configuration
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class GeminiService:
    def __init__(self):
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=1000)
        timeout = httpx.Timeout(60.0, connect=10.0)
        
        self.client = httpx.AsyncClient(
            base_url=settings.GEMINI_BASE_URL,
            timeout=timeout,
            limits=limits,
            follow_redirects=True
        )

    def update_log_level(self, level_name: str):
        """Update logger level dynamically"""
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    async def close(self):
        await self.client.aclose()

    async def get_next_key(self, db: AsyncSession, channel_id: int = None) -> OfficialKey:
        """
        Get the next official key using a round-robin strategy, without checking for active status.
        """
        stmt = select(OfficialKey).options(selectinload(OfficialKey.channel))
        if channel_id:
            stmt = stmt.filter(OfficialKey.channel_id == channel_id)
        
        stmt = stmt.order_by(OfficialKey.id)
        result = await db.execute(stmt)
        keys = result.scalars().all()

        if not keys:
            raise HTTPException(status_code=503, detail=f"No official keys configured{' for this channel' if channel_id else ''}")

        config_result = await db.execute(select(SystemConfig))
        config = config_result.scalars().first()
        if not config:
            config = SystemConfig()
            db.add(config)
        
        last_key_id = config.last_used_official_key_id
        next_key = None

        if last_key_id:
            try:
                last_key_index = next(i for i, key in enumerate(keys) if key.id == last_key_id)
                next_key_index = (last_key_index + 1) % len(keys)
                next_key = keys[next_key_index]
            except StopIteration:
                next_key = keys[0]
        else:
            next_key = keys[0]

        config.last_used_official_key_id = next_key.id
        await db.commit()
        return next_key

    async def get_active_key(self, db: AsyncSession, channel_id: int = None) -> OfficialKey:
        """
        Finds and returns an active OfficialKey object by iterating through available keys.
        """
        stmt = select(OfficialKey).options(selectinload(OfficialKey.channel))
        if channel_id:
            stmt = stmt.filter(OfficialKey.channel_id == channel_id)
        
        result = await db.execute(stmt)
        all_keys = result.scalars().all()
        
        if not all_keys:
            raise HTTPException(status_code=503, detail=f"No official keys configured{' for this channel' if channel_id else ''}")
        
        # Try to find an active key for up to one full loop of all available keys
        for _ in range(len(all_keys)):
            key_obj = await self.get_next_key(db, channel_id=channel_id)
            if key_obj.is_active:
                return key_obj

        raise HTTPException(status_code=503, detail=f"All official keys are disabled{' for this channel' if channel_id else ''}")

    async def get_active_key_str(self, db: AsyncSession, channel_id: int = None) -> str:
        """
        Finds and returns an active key string.
        """
        key_obj = await self.get_active_key(db, channel_id=channel_id)
        return key_obj.key

    async def update_key_status(self, db: AsyncSession, key_str: str, status_code: int, input_tokens: int = 0, output_tokens: int = 0):
        result = await db.execute(select(OfficialKey).filter(OfficialKey.key == key_str))
        key = result.scalars().first()
        if key:
            key.last_status_code = status_code
            key.usage_count += 1

            if 200 <= status_code < 300:
                key.total_tokens = (key.total_tokens or 0) + input_tokens + output_tokens
                key.error_count = 0 # Reset error count on success
                key.last_status = str(status_code)
            else:
                key.error_count = (key.error_count or 0) + 1
                # Auto-disable logic: after 3 consecutive errors
                if key.error_count >= 3:
                    key.is_active = False
                    key.last_status = "auto_disabled"
                else:
                    key.last_status = str(status_code)

            await db.commit()

gemini_service = GeminiService()
