from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel

from .infer_engine import InferEngine

infer_engine = InferEngine()
app = FastAPI()


class VhAndSigma(BaseModel):
    vh: list[list[float]]
    sigma: list[float]


@app.post("/calc_vh_and_sigma")
def calc_vh_and_sigma(z: list[float]) -> VhAndSigma:
    vh, sigma = infer_engine.calc_vh_and_sigma(z)
    return VhAndSigma(vh=vh, sigma=sigma)


handler = Mangum(app)
