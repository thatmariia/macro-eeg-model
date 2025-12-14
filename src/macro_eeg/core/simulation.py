from pydantic import BaseModel, model_validator, Field, ConfigDict
from .nodes import Node
from .connectivity import EdgeConnectivity, EdgeDistance
from .stimuli import Stimulus


class TimeBase(BaseModel):
    sample_rate_hz: int = Field(gt=0)

    @property
    def ms_per_sample(self) -> float:
        return 1000.0 / self.sample_rate_hz

    def samples_to_ms(self, samples: int) -> float:
        return samples * self.ms_per_sample

    def ms_to_samples(self, ms: float) -> int:
        return int(ms / self.ms_per_sample)


class SimulationParams(BaseModel):
    sample_rate_hz: int = Field(gt=0)  # Hz
    lags_ms: int = Field(gt=0)
    sim_ms: int = Field(gt=0)
    burnin_ms: int = Field(ge=0)

    timebase: TimeBase

    @model_validator(mode="before")
    @classmethod
    def inject_timebase(cls, data: dict) -> dict:
        if "timebase" not in data or data["timebase"] is None:
            data["timebase"] = TimeBase(sample_rate_hz=data["sample_rate_hz"])
        return data




class DiameterDist(BaseModel):
    shape: float
    scale: float = Field(gt=0)
    location: float

    @model_validator(mode="after")
    def _check(self):
        if self.shape == 0:
            raise ValueError("shape must not be zero")
        return self
