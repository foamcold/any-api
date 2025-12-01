from typing import Any, List
import hashlib
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api import deps
from app.models.key import OfficialKey, ExclusiveKey
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.schemas.key import OfficialKey as OfficialKeySchema, OfficialKeyCreate, OfficialKeyUpdate, OfficialKeyBatchCreate
from app.schemas.key import ExclusiveKey as ExclusiveKeySchema, ExclusiveKeyCreate, ExclusiveKeyUpdate
from sqlalchemy import func, insert

router = APIRouter()

# --- Official Keys ---

@router.get("/official", response_model=PaginatedResponse[OfficialKeySchema])
async def read_official_keys(
    db: AsyncSession = Depends(deps.get_db),
    page: int = 1,
    size: int = 10,
    status: str = "all", # all, normal, abnormal
    channel_id: int = None,  # 按渠道过滤
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve official keys.
    """
    skip = (page - 1) * size
    query = select(OfficialKey).filter(OfficialKey.user_id == current_user.id)
    
    # 按渠道过滤
    if channel_id is not None:
        query = query.filter(OfficialKey.channel_id == channel_id)
    
    if status == "normal":
        query = query.filter(OfficialKey.is_active == True, (OfficialKey.last_status == "active") | (OfficialKey.last_status == "200"))
    elif status == "abnormal":
        query = query.filter(OfficialKey.is_active == True, (OfficialKey.last_status != "active") & (OfficialKey.last_status != "200"))
    elif status == "manually_disabled":
        query = query.filter(OfficialKey.is_active == False, OfficialKey.last_status != "auto_disabled")
    elif status == "auto_disabled":
        query = query.filter(OfficialKey.is_active == False, OfficialKey.last_status == "auto_disabled")

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)
    
    result = await db.execute(query.offset(skip).limit(size))
    keys = result.scalars().all()
    
    # Manually convert to schema to include last_status_code
    key_schemas = [OfficialKeySchema.from_orm(key) for key in keys]
    
    return PaginatedResponse(
        total=total,
        items=key_schemas,
        page=page,
        size=size
    )

@router.post("/official", response_model=OfficialKeySchema)
async def create_official_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_in: OfficialKeyCreate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new official key, or reuse an existing unlinked key.
    """
    # With the new composite unique constraint, we just need to check
    # if the exact key-user-channel combination already exists.
    # The database will enforce the uniqueness.
    
    result = await db.execute(
        select(OfficialKey).filter_by(
            key=key_in.key,
            user_id=current_user.id,
            channel_id=key_in.channel_id
        )
    )
    existing_key = result.scalar_one_or_none()

    if existing_key:
        raise HTTPException(
            status_code=409,
            detail="此密钥已存在于当前渠道中。",
        )

    # If not exists, create a new one.
    new_key = OfficialKey(
        key=key_in.key,
        user_id=current_user.id,
        is_active=key_in.is_active,
        channel_id=key_in.channel_id,
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)
    return new_key

@router.post("/official/batch")
async def create_official_keys_batch(
    *,
    db: AsyncSession = Depends(deps.get_db),
    keys_in: OfficialKeyBatchCreate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new official keys in batch.
    """
    success_count = 0
    fail_count = 0

    # Get existing keys to avoid duplicates
    existing_keys_res = await db.execute(select(OfficialKey.key).filter(OfficialKey.user_id == current_user.id))
    existing_keys = {row[0] for row in existing_keys_res}

    keys_to_insert = []
    for key_str in keys_in.keys:
        if key_str not in existing_keys:
            keys_to_insert.append({
                "key": key_str,
                "user_id": current_user.id,
                "is_active": keys_in.is_active,
                "channel_id": keys_in.channel_id,
            })
            existing_keys.add(key_str) # Add to set to handle duplicates within the batch
        else:
            fail_count += 1
    
    if keys_to_insert:
        try:
            await db.execute(insert(OfficialKey), keys_to_insert)
            await db.commit()
            success_count = len(keys_to_insert)
        except Exception as e:
            await db.rollback()
            fail_count += len(keys_to_insert)
            print(f"Batch insert failed: {e}")

    return {"success_count": success_count, "fail_count": fail_count}

@router.delete("/official/{key_id}", response_model=OfficialKeySchema)
async def delete_official_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_id: int,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete official key.
    """
    result = await db.execute(select(OfficialKey).filter(OfficialKey.id == key_id, OfficialKey.user_id == current_user.id))
    key = result.scalars().first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    await db.delete(key)
    await db.commit()
    return key

@router.patch("/official/{key_id}", response_model=OfficialKeySchema)
async def update_official_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_id: int,
    key_in: OfficialKeyUpdate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update official key.
    """
    result = await db.execute(select(OfficialKey).filter(OfficialKey.id == key_id, OfficialKey.user_id == current_user.id))
    key = result.scalars().first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    update_data = key_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(key, field, value)
        
    db.add(key)
    await db.commit()
    await db.refresh(key)
    return key

# --- Exclusive Keys ---

@router.get("/exclusive", response_model=PaginatedResponse[ExclusiveKeySchema])
async def read_exclusive_keys(
    db: AsyncSession = Depends(deps.get_db),
    page: int = 1,
    size: int = 10,
    q: str = None,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve exclusive keys.
    """
    skip = (page - 1) * size
    query = select(ExclusiveKey).filter(ExclusiveKey.user_id == current_user.id)
    
    if q:
        query = query.filter(
            (ExclusiveKey.name.ilike(f"%{q}%")) | 
            (ExclusiveKey.key.ilike(f"%{q}%"))
        )
        
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)
        
    result = await db.execute(query.offset(skip).limit(size))
    keys = result.scalars().all()
    
    return PaginatedResponse(
        total=total,
        items=keys,
        page=page,
        size=size
    )

@router.post("/exclusive", response_model=ExclusiveKeySchema)
async def create_exclusive_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_in: ExclusiveKeyCreate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Generate new exclusive key.
    """
    # Generate key logic: gapi- + (sha256前16位 + sha256后16位)
    # SHA256 由 用户id + 用户名 + 时间戳 生成
    timestamp = str(int(time.time()))
    raw_str = f"{current_user.id}{current_user.username}{timestamp}"
    hash_full = hashlib.sha256(raw_str.encode()).hexdigest()
    hash_str = hash_full[:16] + hash_full[-16:]
    generated_key = f"gapi-{hash_str}"
    
    key = ExclusiveKey(
        key=generated_key,
        name=key_in.name,
        user_id=current_user.id,
        is_active=key_in.is_active,
        preset_id=key_in.preset_id,
        channel_id=key_in.channel_id,
        enable_regex=key_in.enable_regex,
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)
    return key

@router.delete("/exclusive/{key_id}", response_model=ExclusiveKeySchema)
async def delete_exclusive_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_id: int,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete exclusive key.
    """
    result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.id == key_id, ExclusiveKey.user_id == current_user.id))
    key = result.scalars().first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    await db.delete(key)
    await db.commit()
    return key

@router.patch("/exclusive/{key_id}", response_model=ExclusiveKeySchema)
async def update_exclusive_key(
    *,
    db: AsyncSession = Depends(deps.get_db),
    key_id: int,
    key_in: ExclusiveKeyUpdate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update exclusive key.
    """
    result = await db.execute(select(ExclusiveKey).filter(ExclusiveKey.id == key_id, ExclusiveKey.user_id == current_user.id))
    key = result.scalars().first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    update_data = key_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(key, field, value)
        
    db.add(key)
    await db.commit()
    await db.refresh(key)
    return key
