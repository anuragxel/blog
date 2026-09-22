from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    size: int = 96
    patch: int = 8
    width: int = 192
    depth: int = 6
    heads: int = 3
    dim: int = 128
    batch: int = 64
    steps: int = 1000
    lr: float = 0.0003
    warmup: int = 100
    wd: float = 0.04
    momentum: float = 0.996
    eps: float = 0.5
    gamma: float = 0.02
    bf16: bool = False
