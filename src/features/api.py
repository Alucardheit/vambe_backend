from fastapi import APIRouter

from .clients.endpoints import router as clients_router
from .indicators.endpoints import router as indicators_router

router = APIRouter()

router.include_router(clients_router, prefix="/clients", tags=["clients"])
router.include_router(indicators_router, prefix="/indicators", tags=["indicators"])
