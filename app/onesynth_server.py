import json
import sys
import threading
import time

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.infer_engine import InferEngine

infer_engine = InferEngine()
app = FastAPI()


class CommonRequest(BaseModel):
    position: float
    basis: list[float]
    center_z: list[float]


class SeekResponse(BaseModel):
    synth_params: list[float]


class MoveForwardResponse(BaseModel):
    vh: list[list[float]]
    sigma: list[float]
    new_center_z: list[float]


class TurnAroundRequest(BaseModel):
    vh: list[list[float]]
    sigma: list[float]


class TurnAroundResponse(BaseModel):
    basis: list[float]


async def parse_body(request: Request) -> dict:
    data: bytes = await request.body()
    return json.loads(data.decode("utf-8"))


@app.post("/move_forward")
def move_forward(data: dict = Depends(parse_body)) -> MoveForwardResponse:
    req = CommonRequest(**data)
    z = infer_engine.calc_z(req.position, req.basis, req.center_z)
    vh, sigma = infer_engine.calc_vh_and_sigma(z)
    return MoveForwardResponse(vh=vh, sigma=sigma, new_center_z=z)


@app.post("/seek")
def seek(data: dict = Depends(parse_body)) -> SeekResponse:
    req = CommonRequest(**data)
    z = infer_engine.calc_z(req.position, req.basis, req.center_z)
    return SeekResponse(synth_params=infer_engine.infer_synth_params(z))


@app.post("/turn_around")
def turn_around(data: dict = Depends(parse_body)) -> TurnAroundResponse:
    req = TurnAroundRequest(**data)
    return TurnAroundResponse(basis=infer_engine.turn_around(req.vh, req.sigma))


keep_alive = True


@app.get("/ping")
def ping() -> str:
    global keep_alive
    keep_alive = True
    return "ok"


def auto_exit() -> None:
    global keep_alive
    while True:
        if not keep_alive:
            sys.exit()
        keep_alive = False
        time.sleep(60 * 5)


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


if __name__ == "__main__":
    main_thread = threading.Thread(target=run)
    main_thread.daemon = True
    main_thread.start()
    auto_exit()
