from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.monitoring import router as monitoring_router
from app.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.include_router(health_router)
app.include_router(monitoring_router)
