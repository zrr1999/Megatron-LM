# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout- and partition-independent L2 norm of FP32 gradients.

Squares are computed in FP32 on the gradient device. Their mantissas are
accumulated into base-65536 integer bins, which may be SUM-reduced across
owners before rounding the total once to FP32 and applying the native sqrt.
No gradient values or floating-point norm computation leave the device.

The 20 limbs cover FP32 squares and the supported 2**40 global element count.
Each uncarried limb is bounded by 2**40 * 65535 < 2**56, so arbitrary parameter
and rank reduction orders cannot overflow int64. The last three bins count
infinities, NaNs, and elements. This deliberately costs more than a fused norm
and is intended only for explicitly enabled accuracy compatibility.
"""

import torch


class ReproducibleL2Norm:
    """Accumulate on a single device; reduce bins before calling ``finish``."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = (
            device if device is not None else torch.device("cuda", torch.cuda.current_device())
        )

    def cast(self, value: torch.Tensor, dtype: str) -> torch.Tensor:
        return value.to(getattr(torch, dtype))

    def zeros(self, count: int = 23) -> torch.Tensor:
        return torch.zeros(count, dtype=torch.int64, device=self.device)

    def tensor(
        self, value: list[int | float] | int | float, dtype: str = "float32"
    ) -> torch.Tensor:
        return torch.tensor(value, dtype=getattr(torch, dtype), device=self.device)

    def view(self, value: torch.Tensor, dtype: str) -> torch.Tensor:
        return value.view(getattr(torch, dtype))

    def add(self, bins: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return bins.scatter_add_(0, indices, values)

    def accumulate(
        self, bins: torch.Tensor, gradient: torch.Tensor, chunk_size: int = 1048576
    ) -> torch.Tensor:
        if gradient.layout != torch.strided or gradient.dtype != torch.float32:
            raise TypeError("Reproducible clipping requires FP32 gradients")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        flat = gradient.reshape([-1])
        if flat.shape[0] > 2**40:
            raise OverflowError("Reproducible norm supports at most 2**40 global elements")
        bins = self.add(bins, self.tensor([22], "int64"), self.tensor([flat.shape[0]], "int64"))
        for start in range(0, flat.shape[0], chunk_size):
            value = flat[start : start + chunk_size]
            bits = self.cast(self.view(value * value, "int32"), "int64")
            exponent = (bits >> 23) & self.tensor(255, "int64")
            fraction = bits & self.tensor(0x7FFFFF, "int64")
            finite = exponent != 255
            mantissa = fraction | torch.where(
                exponent > 0, torch.full_like(exponent, 0x800000), torch.zeros_like(exponent)
            )
            shift = torch.maximum(exponent - 1, torch.zeros_like(exponent))
            limb = shift // 16
            shifted = (mantissa << (shift % 16)) * self.cast(finite, "int64")
            for offset in range(3):
                bins = self.add(
                    bins, limb + offset, (shifted >> (16 * offset)) & self.tensor(65535, "int64")
                )
            flags = torch.stack(
                [
                    self.cast((exponent == 255) & (fraction == 0), "int64").sum(),
                    self.cast((exponent == 255) & (fraction != 0), "int64").sum(),
                ]
            )
            bins = self.add(bins, self.tensor([20, 21], "int64"), flags)
        return bins

    def finish(self, bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(bins[22].item()) > 2**40:
            raise OverflowError("Reproducible norm supports at most 2**40 global elements")
        carry = self.zeros(1)[0]
        digits = []
        for i in range(20):
            value = bins[i] + carry
            digits.append(value & self.tensor(65535, "int64"))
            carry = value >> 16
        digits = torch.stack(digits)
        index = self.tensor(list(range(20)), "int64")
        top = torch.where(digits != 0, index, torch.full_like(index, -1)).max()

        def get(i):
            return torch.where(index == i, digits, torch.zeros_like(digits)).sum()

        word = get(top)
        leading = self.zeros(1)[0]
        for width in [8, 4, 2, 1]:
            take = word >= (1 << width)
            word = torch.where(take, word >> width, word)
            leading = leading + self.cast(take, "int64") * width
        highest = top * 16 + leading
        cut = torch.maximum(highest - 23, self.zeros(1)[0])
        limb, shift = cut // 16, cut % 16
        significand = (
            (get(limb) >> shift) | (get(limb + 1) << (16 - shift)) | (get(limb + 2) << (32 - shift))
        ) & self.tensor(0xFFFFFF, "int64")
        round_position = torch.maximum(cut - 1, self.zeros(1)[0])
        round_limb, round_shift = round_position // 16, round_position % 16
        round_word = get(round_limb)
        round_bit = ((round_word >> round_shift) & self.tensor(1, "int64")) * self.cast(
            cut > 0, "int64"
        )
        sticky = torch.where(index < round_limb, digits, torch.zeros_like(digits)).sum() != 0
        sticky = sticky | ((round_word & ((self.tensor(1, "int64") << round_shift) - 1)) != 0)
        significand = significand + round_bit * self.cast(
            sticky | ((significand & self.tensor(1, "int64")) != 0), "int64"
        )
        exponent = highest - 22 + self.cast(significand == 0x1000000, "int64")
        raw = (exponent << 23) | (significand & self.tensor(0x7FFFFF, "int64"))
        raw = torch.where(exponent >= 255, torch.full_like(raw, 0x7F800000), raw)
        raw = torch.where(highest < 23, get(0) | (get(1) << 16), raw)
        raw = torch.where(top < 0, torch.zeros_like(raw), raw)
        raw = torch.where(bins[20] != 0, torch.full_like(raw, 0x7F800000), raw)
        raw = torch.where(bins[21] != 0, torch.full_like(raw, 0x7FC00000), raw)
        square_sum = self.view(self.cast(raw.reshape([1]), "int32"), "float32")
        return torch.sqrt(square_sum), square_sum
