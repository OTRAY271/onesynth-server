import json

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
    prob: list[float]
    new_center_z: list[float]


class TurnAroundRequest(BaseModel):
    vh: list[list[float]]
    prob: list[float]


class TurnAroundResponse(BaseModel):
    basis: list[float]


async def parse_body(request: Request) -> dict:
    data: bytes = await request.body()
    return json.loads(data.decode("utf-8"))


@app.post("/move_foward")
def move_forward(data: dict = Depends(parse_body)) -> MoveForwardResponse:
    req = CommonRequest(**data)
    z = infer_engine.calc_z(req.position, req.basis, req.center_z)
    vh, sigma = infer_engine.calc_vh_and_sigma(z)
    prob = infer_engine.calc_prob(sigma)
    return MoveForwardResponse(vh=vh, prob=prob, new_center_z=z)


@app.post("/seek")
def seek(data: dict = Depends(parse_body)) -> SeekResponse:
    req = CommonRequest(**data)
    z = infer_engine.calc_z(req.position, req.basis, req.center_z)
    return SeekResponse(synth_params=infer_engine.infer_synth_params(z))


@app.post("/turn_around")
def turn_around(data: dict = Depends(parse_body)) -> TurnAroundResponse:
    req = TurnAroundRequest(**data)
    return TurnAroundResponse(basis=infer_engine.turn_around(req.vh, req.prob))


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
