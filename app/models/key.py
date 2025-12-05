from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, BigInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

from sqlalchemy import UniqueConstraint
from typing import List, Dict, Any

# --- Models ---

class OfficialKey(Base):
    __tablename__ = "official_keys"
    __table_args__ = (
        UniqueConstraint('key', 'user_id', 'channel_id', name='_user_channel_key_uc'),
    )

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=True)
    usage_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    input_tokens = Column(BigInteger, default=0)
    output_tokens = Column(BigInteger, default=0)
    last_status = Column(String, default="active")
    last_status_code = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    channel = relationship("Channel")

class ExclusiveKey(Base):
    __tablename__ = "exclusive_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False) # gapi-...
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    preset_id = Column(Integer, ForeignKey("presets.id"), nullable=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=True)
    enable_regex = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    preset = relationship("Preset")
    channel = relationship("Channel")
