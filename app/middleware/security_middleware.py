from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全响应头中间件
    
    为所有响应添加安全相关的HTTP头，包括：
    - Content-Security-Policy: 允许Cloudflare Turnstile等外部资源
    - X-Content-Type-Options: 防止MIME类型嗅探
    - X-Frame-Options: 防止点击劫持
    - X-XSS-Protection: 启用XSS过滤
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 设置Content-Security-Policy
        # 允许Cloudflare Turnstile所需的资源
        # 使用通配符以支持所有Cloudflare子域名
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://challenges.cloudflare.com https://*.cloudflare.com https://*.cloudflareinsights.com; "
            "frame-src 'self' https://challenges.cloudflare.com https://*.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://*.cloudflare.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://challenges.cloudflare.com https://*.cloudflare.com https://*.cloudflareinsights.com;"
        )
        
        response.headers["Content-Security-Policy"] = csp_policy
        
        # 其他安全响应头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response
