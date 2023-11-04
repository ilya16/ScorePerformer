from dataclasses import dataclass


@dataclass
class Note:
    pitch: int
    velocity: int
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start
