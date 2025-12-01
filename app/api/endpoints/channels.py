from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.api import deps
from app.models.user import User
from app.models.channel import Channel as ChannelModel
from app.models.key import OfficialKey
from app.schemas.channel import Channel as ChannelSchema, ChannelCreate, ChannelUpdate

router = APIRouter()

@router.get("/", response_model=List[ChannelSchema])
async def read_channels(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取当前用户的渠道列表
    """
    result = await db.execute(
        select(ChannelModel).where(ChannelModel.user_id == current_user.id)
    )
    channels = result.scalars().all()
    return channels

@router.post("/", response_model=ChannelSchema)
async def create_channel(
    *,
    db: AsyncSession = Depends(deps.get_db),
    channel_in: ChannelCreate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    创建新渠道
    """
    channel = ChannelModel(
        name=channel_in.name,
        type=channel_in.type,
        api_url=channel_in.api_url,
        user_id=current_user.id
    )
    db.add(channel)
    await db.commit()
    await db.refresh(channel)
    return channel

@router.patch("/{channel_id}", response_model=ChannelSchema)
async def update_channel(
    *,
    db: AsyncSession = Depends(deps.get_db),
    channel_id: int,
    channel_in: ChannelUpdate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新渠道信息
    """
    result = await db.execute(
        select(ChannelModel).where(
            ChannelModel.id == channel_id,
            ChannelModel.user_id == current_user.id
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(status_code=404, detail="渠道不存在")
    
    update_data = channel_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(channel, field, value)
    
    await db.commit()
    await db.refresh(channel)
    return channel

@router.delete("/{channel_id}")
async def delete_channel(
    *,
    db: AsyncSession = Depends(deps.get_db),
    channel_id: int,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    删除渠道
    """
    result = await db.execute(
        select(ChannelModel).where(
            ChannelModel.id == channel_id,
            ChannelModel.user_id == current_user.id
        )
    )
    channel = result.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(status_code=404, detail="渠道不存在")
    
    # Unlink official keys
    result = await db.execute(
        select(OfficialKey).where(OfficialKey.channel_id == channel_id)
    )
    for key in result.scalars().all():
        key.channel_id = None
        key.is_active = False # Deactivate key when channel is deleted
        
    await db.delete(channel)
    await db.commit()
    
    return {"message": "渠道已删除"}
