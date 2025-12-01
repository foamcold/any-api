from sqlalchemy import Column, Integer, String, Boolean, Text
from app.core.database import Base

class SystemConfig(Base):
    """系统配置模型"""
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    
    # 基本设置
    site_name = Column(String, default="Any API")
    server_url = Column(String, nullable=True, default=None)
    
    # 注册配置
    allow_registration = Column(Boolean, default=True)
    allow_password_login = Column(Boolean, default=True)
    require_email_verification = Column(Boolean, default=False)
    enable_turnstile = Column(Boolean, default=False)
    enable_captcha = Column(Boolean, default=False)
    enable_ip_rate_limit = Column(Boolean, default=False)
    
    # 邮箱配置
    email_whitelist_enabled = Column(Boolean, default=False)
    email_whitelist = Column(Text, default='["qq.com", "outlook.com", "gmail.com"]')  # JSON数组
    
    # SMTP配置
    smtp_host = Column(String, nullable=True)
    smtp_port = Column(Integer, default=587)
    smtp_user = Column(String, nullable=True)
    smtp_password = Column(String, nullable=True)
    smtp_from = Column(String, nullable=True)
    smtp_use_tls = Column(Boolean, default=True)
    
    # Turnstile配置
    turnstile_site_key = Column(String, nullable=True)
    turnstile_secret_key = Column(String, nullable=True)

    # 日志配置
    log_level = Column(String, default="INFO") # DEBUG, INFO, WARNING, ERROR

    # 密钥轮询配置
    last_used_official_key_id = Column(Integer, nullable=True)
