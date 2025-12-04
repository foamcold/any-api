from pydantic import BaseModel
from typing import Optional, List

class SystemConfigBase(BaseModel):
    """系统配置基础schema"""
    site_name: str = "Any API"
    server_url: Optional[str] = None
    
    # 注册配置
    allow_registration: bool = True
    allow_password_login: bool = True
    require_email_verification: bool = False
    enable_turnstile: bool = False
    enable_captcha: bool = False
    enable_ip_rate_limit: bool = False
    
    # 邮箱配置
    email_whitelist_enabled: bool = False
    email_whitelist: List[str] = ["qq.com", "outlook.com", "gmail.com"]
    
    # SMTP配置
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Turnstile配置
    turnstile_site_key: Optional[str] = None
    turnstile_secret_key: Optional[str] = None

    # 日志配置
    log_level: str = "INFO"

    # 伪流配置
    pseudo_streaming_enabled: bool = True

class SystemConfigCreate(SystemConfigBase):
    """创建系统配置"""
    pass

class SystemConfigUpdate(SystemConfigBase):
    """更新系统配置"""
    pass

class SystemConfig(SystemConfigBase):
    """系统配置完整schema"""
    id: int
    
    class Config:
        from_attributes = True
