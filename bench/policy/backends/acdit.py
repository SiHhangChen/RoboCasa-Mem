"""Reserved AC-DiT backend boundary.

AC-DiT requires the Phase 5 RGB-D/history contract and is intentionally not
silently mapped onto the RGB-only v1 implementation.
"""


class ACDiTBackend:
    @classmethod
    def from_checkpoint(cls, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "AC-DiT serving requires the RGB-D/history Phase 5 implementation"
        )
