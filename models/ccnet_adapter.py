"""CCNet adapter.

The released checkpoint contains 36 ``att_*`` tensors absent from the official
forward graph. ``configs/ccnet.yaml`` declares them as ignored and asserts their
exact count; every other tensor is loaded strictly.
"""
from models.model_adapter import PalmprintModelAdapter


class CCNetAdapter(PalmprintModelAdapter):
    name = "ccnet"
