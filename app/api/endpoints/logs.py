from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from app.api import deps
from app.models.log import Log
from app.models.user import User
from pydantic import BaseModel
from datetime import datetime, timezone
from app.schemas.common import PaginatedResponse

router = APIRouter()

class LogSchema(BaseModel):
    id: int
    model: str
    status: str
    status_code: Optional[int]
    latency: float
    ttft: float
    is_stream: bool
    input_tokens: int
    output_tokens: int
    created_at: datetime
    exclusive_key_key: Optional[str] = None
    official_key_key: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
        }

@router.get("/", response_model=PaginatedResponse[LogSchema])
async def read_logs(
    db: AsyncSession = Depends(deps.get_db),
    page: int = 1,
    size: int = 20,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve logs.
    """
    skip = (page - 1) * size
    if current_user.role in ["admin", "super_admin"]:
        base_query = select(Log)
    else:
        base_query = select(Log).filter(Log.user_id == current_user.id)
    
    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total = await db.scalar(count_query)

    # Get paginated results
    query = base_query.order_by(Log.created_at.desc()).options(
        selectinload(Log.exclusive_key),
        selectinload(Log.official_key)
    ).offset(skip).limit(size)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    results = []
    for log in logs:
        log_data = {
            "id": log.id,
            "model": log.model,
            "status": log.status,
            "status_code": log.status_code,
            "latency": log.latency,
            "ttft": log.ttft,
            "is_stream": log.is_stream,
            "input_tokens": log.input_tokens,
            "output_tokens": log.output_tokens,
            "created_at": log.created_at,
            "exclusive_key_key": log.exclusive_key.key if log.exclusive_key else None,
            "official_key_key": log.official_key.key if log.official_key else None
        }
        results.append(LogSchema(**log_data))
        
    return PaginatedResponse(
        total=total,
        items=results,
        page=page,
        size=size
    )
