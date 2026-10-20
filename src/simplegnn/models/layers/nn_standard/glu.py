import math

import torch
from torch import nn

from simplegnn.models.layers.framework_layer import FrameworkLayer


class GLULayer(FrameworkLayer):
    # The GLU family: the gate activation is the only thing that distinguishes
    # the variants, so the named variants are presets of gate_activation.
    GATE_ACTIVATIONS = {
        'silu': torch.nn.functional.silu,        # SwiGLU
        'gelu': torch.nn.functional.gelu,        # GEGLU
        'relu': torch.nn.functional.relu,        # ReGLU
        'sigmoid': torch.sigmoid,                # GLU (Dauphin et al.)
        'identity': lambda x: x,                 # Bilinear
    }
    VARIANT_GATES = {
        'swiglu': 'silu',
        'geglu': 'gelu',
        'reglu': 'relu',
        'glu': 'sigmoid',
        'bilinear': 'identity',
    }

    def __init__(self, layer_args):
        """
        A gated linear unit (GLU) feed-forward block: two parallel projections
        of the input, one of them passed through the gate activation, are
        multiplied elementwise and projected back down:

            out = (gate(x W_g + b_g) * (x W_u + b_u)) W_d + b_d

        The gate activation is what distinguishes the GLU variants, so it is a
        parameter: 'silu' gives SwiGLU (the default), 'sigmoid' the original
        GLU of Dauphin et al., 'gelu' GEGLU, 'relu' ReGLU and 'identity' the
        bilinear variant. The layer_type 'swiglu' is accepted as an alias of
        'glu', since that is the name the default variant is usually
        configured under.

        The gate and the up projection both map in_features -> hidden_features,
        the down projection maps hidden_features -> out_features. The block only
        touches the last (feature) dimension, so it accepts every tensor shape
        used in the framework: (N, F) node-level, (C, N, F) multi-head and
        (B, C, F) batched graph-level inputs all keep their leading dimensions
        and their channel count.

        Optional layer_args keys:
        - 'variant' : str -> named preset for the gate, one of 'swiglu',
          'geglu', 'reglu', 'glu', 'bilinear' (default: 'swiglu')
        - 'gate_activation' : str -> the gate directly, one of 'silu', 'gelu',
          'relu', 'sigmoid', 'identity'. Takes precedence over 'variant'.
        - 'hidden_features' : int -> width of the gated inner projection
          (default: expansion * in_features)
        - 'expansion' : float (default 2.0) -> only used when hidden_features is
          not given. The gated block has three weight matrices instead of the
          two of a plain MLP, so the usual transformer choice is to shrink the
          inner width by 2/3 (expansion 8/3 instead of 4) to keep the parameter
          count comparable.
        - 'out_features' : int (default: in_features) -> a feed-forward block is
          usually dimension preserving
        - 'bias' : bool (default True) -> biases on all three projections
        - 'residual' : bool (default False) -> requires in_features == out_features
        - 'dropout' : float (default 0.0) -> applied to the block output
        """
        layer_args['name'] = "GLU Layer"
        super(GLULayer, self).__init__(layer_args=layer_args)
        variant = layer_args.get('variant', None)
        if variant is not None and variant not in self.VARIANT_GATES:
            raise ValueError(f"Unsupported glu variant: '{variant}'. "
                             f"Supported variants: {list(self.VARIANT_GATES.keys())}.")
        default_gate = 'silu' if variant is None else self.VARIANT_GATES[variant]
        self.gate_activation_name = layer_args.get('gate_activation', default_gate)
        if self.gate_activation_name not in self.GATE_ACTIVATIONS:
            raise ValueError(f"Unsupported glu gate activation: '{self.gate_activation_name}'. "
                             f"Supported values: {list(self.GATE_ACTIVATIONS.keys())}.")
        self.gate_activation = self.GATE_ACTIVATIONS[self.gate_activation_name]
        self.name = f"GLU Layer ({self.gate_activation_name} gate)"
        self.expansion = layer_args.get('expansion', 2.0)
        self.hidden_features = int(layer_args.get('hidden_features', round(self.expansion * self.in_features)))
        if self.hidden_features <= 0:
            raise ValueError(f"hidden_features must be positive, got {self.hidden_features}")
        if self.residual and self.in_features != self.out_features:
            raise ValueError(f"residual requires in_features == out_features, but got "
                             f"in_features={self.in_features} and out_features={self.out_features}")
        torch.manual_seed(self.layer_id + self.seed)
        # nn.Linear-style uniform initialization, fan_in of the respective projection
        k = math.sqrt(1.0 / self.in_features)
        self.Param_W_g = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.hidden_features, dtype=self.precision), -k, k))
        self.Param_W_u = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.hidden_features, dtype=self.precision), -k, k))
        d = math.sqrt(1.0 / self.hidden_features)
        self.Param_W_d = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.hidden_features, self.out_features, dtype=self.precision), -d, d))
        if self.bias:
            self.Param_b_g = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.hidden_features, dtype=self.precision), -k, k))
            self.Param_b_u = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.hidden_features, dtype=self.precision), -k, k))
            self.Param_b_d = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -d, d))

    def forward(self, node_representation: torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer
        param: node_representation: torch.Tensor -> the input tensor, (..., F)
        """
        x = node_representation
        gate = node_representation @ self.Param_W_g
        up = node_representation @ self.Param_W_u
        if self.bias:
            gate = gate + self.Param_b_g
            up = up + self.Param_b_u
        out = (self.gate_activation(gate) * up) @ self.Param_W_d
        if self.bias:
            out = out + self.Param_b_d
        if self.batch_norm:
            out = self.batch_norm_layer(out)
        out = self.activation(out)
        if self.residual:
            out = out + x
        if self.dropout > 0 and self.training:
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
        return out
