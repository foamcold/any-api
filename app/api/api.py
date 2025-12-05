from fastapi import APIRouter
from app.api.endpoints import auth, users, keys, channels, presets, regex, preset_regex, proxy, logs, system, setup, captcha

api_router = APIRouter()
api_router.include_router(setup.router, prefix="/setup", tags=["setup"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(keys.router, prefix="/keys", tags=["keys"])
api_router.include_router(channels.router, prefix="/channels", tags=["channels"])
api_router.include_router(presets.router, prefix="/presets", tags=["presets"])
api_router.include_router(regex.router, prefix="/regex", tags=["regex"])
api_router.include_router(preset_regex.router, tags=["preset_regex"])  # 预设内部正则
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(captcha.router, prefix="/captcha", tags=["captcha"])
# api_router.include_router(gemini_routes.router, tags=["gemini_routes"]) # 移至 main.py 根路径挂载
# api_router.include_router(generic_proxy.router, tags=["generic_proxy"]) # 移至 main.py 根路径挂载

