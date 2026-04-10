from fastapi import APIRouter, Depends
from app.api.deps import require_api_key
from app.api.routes.health import router as health_router
from app.api.routes.slides import collections_router, router as slides_router
from app.api.routes.inference import router as inference_router
from app.api.routes.training import router as training_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(slides_router, dependencies=[Depends(require_api_key)])
api_router.include_router(collections_router, dependencies=[Depends(require_api_key)])
api_router.include_router(inference_router, dependencies=[Depends(require_api_key)])
api_router.include_router(training_router, dependencies=[Depends(require_api_key)])
