from typing import Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime

# Channel Schemas
class ChannelBase(BaseModel):
    name: str
    type: str  # gemini, openai, etc.
    api_url: str

class ChannelCreate(ChannelBase):
    pass

class ChannelUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    api_url: Optional[str] = None

class Channel(ChannelBase):
    id: int
    user_id: int
    created_at: datetime
    # 聚合统计信息
    total_keys: int = 0
    active_keys: int = 0
    usage_count: int = 0
    error_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    class Config:
        from_attributes = True
