# 启动脚本，确保从 .env 加载配置

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    from app.main import app

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        # 如果需要，可以在这里添加其他 uvicorn 参数
        # 例如: reload=True
    )