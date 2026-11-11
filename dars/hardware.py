"""Parameter template and explicitly uncertain PPA extrapolations."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Hardware:
    mhz: float = 100
    gemv_bytes: int = 128
    gemv_batch: int = 4
    vpu_lanes: int = 4
    fdot_lanes: int = 16
    sfu_lanes: int = 1
    read_gbs: float = 12.79
    write_bytes: int = 16
    hold_kib: int = 4
    outstanding: int = 128
    queue_depth: int = 32
    mxu_rows: int = 0
    mxu_cols: int = 0

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Hardware parameters must be finite")
        if any(v <= 0 for k, v in asdict(self).items() if not k.startswith("mxu_")):
            raise ValueError("Hardware parameters must be positive")
        if any(v != int(v) for k, v in asdict(self).items() if k not in ("mhz", "read_gbs")):
            raise ValueError("Hardware widths, counts and capacities must be integers")
        if min(self.mxu_rows, self.mxu_cols) < 0 or bool(self.mxu_rows) != bool(self.mxu_cols):
            raise ValueError("MXU dimensions must both be zero or both positive")


BASE = Hardware()
LIMITS = dict(lut=117120, ff=234240, dsp=1248, bram=144, uram=64)


def ppa(h, target="fpga"):
    # Intercepts anchored to v3.3 total resources, slopes are engineering priors,
    # NOT measured per-unit areas. MXU prior is of the order of the reported removal.
    lut = (104879 + 110 * (h.gemv_bytes - 128) + 1450 * (h.vpu_lanes - 4)
           + 350 * (h.fdot_lanes - 16) + 1800 * (h.sfu_lanes - 1)
           + 20 * (h.outstanding - 128) + 15 * (h.queue_depth - 32)
           + 80 * (h.hold_kib - 4) + 700 * (h.gemv_batch - 4)
           + 40 * (h.write_bytes - 16))
    cells = h.mxu_rows * h.mxu_cols
    lut += (8000 + 500 * cells) if cells else 0
    dsp = (903 + (h.gemv_bytes - 128) * 2 + (h.fdot_lanes - 16) * 6
           + (h.vpu_lanes - 4) * 12 + cells + h.gemv_bytes / 16 * (h.gemv_batch - 4))
    bram = 118 + math.ceil((h.hold_kib - 4) / 4.5) + math.ceil((h.queue_depth - 32) / 64) + math.ceil(cells / 32)
    ff = 59170 * lut / 105992  # v3.4 FF observation, not a v3.3 measurement
    ratio = max(0.1, (lut / 104879 + dsp / 903 + bram / 118) / 3)
    # v3.4 Vivado estimate (not board power): static + PS8 + scaled PL dynamic.
    watts = 0.312 + 2.435 + (4.131 - 2.435) * ratio * h.mhz / 100
    values = dict(lut=lut, ff=ff, dsp=dsp, bram=bram, uram=0)
    feasible = all(0 <= values[k] <= cap for k, cap in LIMITS.items())
    if target == "fpga":
        return dict(**values, area=lut / LIMITS["lut"], power_w=watts, feasible=feasible,
                    status="baseline-anchored resource estimate; slopes and timing unvalidated")
    if target != "asic":
        raise ValueError("target must be fpga or asic")
    # Illustrative 28nm-equivalent assumptions, no technology library or synthesis.
    area = max(0.01, lut * 0.000008 + dsp * 0.00035 + bram * 0.036864 * 0.12)
    power = max(0.01, 0.08 * area + 0.002 * area * h.mhz)
    return dict(area=area, area_mm2=area, power_w=power, feasible=True,
                status="UNCALIBRATED illustrative 28nm-equivalent ASIC estimate")
