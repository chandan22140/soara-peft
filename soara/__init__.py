"""
SOARA: Subspace Orthogonal Adaptation via Rotational Alignment
==============================================================

A family of parameter-efficient fine-tuning (PEFT) methods that adapt pretrained
models by learning lightweight rotational transformations within SVD subspaces.

Quick Start::

    from soara import SOARAConfig, replace_linear_with_soara
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    config = SOARAConfig(r=16, method="v1")
    replace_linear_with_soara(model, config, target_modules=["q_proj", "v_proj", "k_proj"])
    # Model is now SOARAed — train as usual!

Variants:
    - **SOARA-V1**: Dense rotations with orthogonality regularization
    - **SOARA-V2a**: Exact orthogonality via sequential Givens rotations
    - **SOARA-V2b**: Exact orthogonality via butterfly factorizations
"""

from .soara_layer import SOARAConfig, SOARALinearLayer, replace_linear_with_soara

__version__ = "0.1.0"
__all__ = ["SOARAConfig", "SOARALinearLayer", "replace_linear_with_soara"]
