from pydantic import BaseModel, Field


class TimeBase(BaseModel):
    sample_rate_hz: int = Field(gt=0)

    @property
    def ms_per_sample(self) -> float:
        return 1000.0 / self.sample_rate_hz

    def samples_to_ms(self, samples: int) -> float:
        return samples * self.ms_per_sample

    def ms_to_samples(self, ms: float) -> int:
        return int(ms / self.ms_per_sample)
