"""FORESIGHT scoring API.  Run:  uvicorn api.main:app --port 8000   (docs at /docs)"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import forecasts, health, inventory
from src.utils.logging_config import get_logger

log = get_logger("foresight.api")
app = FastAPI(title="FORESIGHT Scoring API", version="1.0.0",
              description="Forecast and inventory scoring using the same business logic as the dashboard. Stateless: no data is stored.")
app.include_router(health.router)
app.include_router(forecasts.router)
app.include_router(inventory.router)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal error while scoring. Check the request and server logs."})
