from pydantic import BaseModel, model_validator, Field, ConfigDict
from .nodes import Node
from .connectivity import EdgeConnectivity, EdgeDistance
from .stimuli import Stimulus


class SimulationParams(BaseModel):
    sample_rate_hz: int = Field(gt=0)  # Hz
    lags_ms: int = Field(gt=0)
    sim_ms: int = Field(gt=0)
    burnin_ms: int = Field(ge=0)


class DiameterDist(BaseModel):
    shape: float
    scale: float = Field(gt=0)
    location: float

    @model_validator(mode="after")
    def _check(self):
        if self.shape == 0:
            raise ValueError("shape must not be zero")
        return self
