import math

import torch
from torch import nn

from simplegnn.models.ShareGNN.utils import is_batched_pos
from simplegnn.models.layers.framework_layer import FrameworkLayer


class AttentionReadoutLayer(FrameworkLayer):
    def __init__(self, layer_args):
        """
        An attention-based graph-level readout over the head axis of an
        unflattened aggregation output. The input tensor is expected to have
        the shape (H, 1, F) per graph or (B, H, F) in the batched forward,
        where H is the number of aggregation heads (reported as channels by
        ``invariant_based_aggregation`` with ``flatten: False``) and F is the
        number of features per head. The H head embeddings are treated as a
        set of tokens and pooled into a single (B, F') graph representation,
        so the layer replaces the dense (H*F, F') readout linear.

        The mechanism depends on the variant parameter:
            - 'gated': gated attention pooling. Each token is scored with
              w . (tanh(x V) * sigmoid(x U)), the scores are softmaxed over H
              and the weighted sum is projected to out_features.
            - 'pma': num_seeds learned query vectors cross-attend over the
              tokens via torch.nn.MultiheadAttention; the seed outputs are
              flattened and projected to out_features.
            - 'transformer': one pre-LN transformer encoder block over the
              tokens (self-attention + feed-forward), followed by the gated
              attention pooling of the 'gated' variant.
        If head_embeddings is True (default), a learned (H, F) embedding
        table is added to the tokens first: attention is otherwise
        permutation-invariant over the tokens, so without it the identity of
        the aggregation head a token came from would only be implicit in its
        activations.
        Default variant is 'gated'.
        """
        layer_args['name'] = "Attention Readout Layer"
        super(AttentionReadoutLayer, self).__init__(layer_args=layer_args)
        self.variant = layer_args.get('variant', 'gated')
        if self.variant not in ('gated', 'pma', 'transformer'):
            raise ValueError(f"Unsupported attention readout variant: '{self.variant}'. "
                             f"Supported variants: 'gated', 'pma', 'transformer'.")
        self.attention_dim = layer_args.get('attention_dim', 64)
        self.num_seeds = layer_args.get('num_seeds', 1)
        self.num_attention_heads = layer_args.get('num_attention_heads', 4)
        self.head_embeddings = layer_args.get('head_embeddings', True)
        self.ffn_dim = layer_args.get('ffn_dim', 2 * self.in_features)
        if self.variant in ('pma', 'transformer') and self.in_features % self.num_attention_heads != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by "
                             f"num_attention_heads ({self.num_attention_heads}) for variant '{self.variant}'")
        # the token count is the head axis of the incoming tensor
        self.num_tokens = self.in_channels
        torch.manual_seed(self.layer_id + self.seed)
        if self.head_embeddings:
            self.Param_E = nn.Parameter(torch.zeros(self.num_tokens, self.in_features, dtype=self.precision).normal_(std=0.02))
        if self.variant == 'transformer':
            self.encoder = nn.TransformerEncoderLayer(d_model=self.in_features, nhead=self.num_attention_heads,
                                                      dim_feedforward=self.ffn_dim, dropout=0.0,
                                                      activation='gelu', batch_first=True,
                                                      norm_first=True).to(self.precision)
        if self.variant in ('gated', 'transformer'):
            k = math.sqrt(1.0 / self.in_features)
            self.Param_V = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.attention_dim, dtype=self.precision), -k, k))
            self.Param_U = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.attention_dim, dtype=self.precision), -k, k))
            d = math.sqrt(1.0 / self.attention_dim)
            self.Param_w = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.attention_dim, dtype=self.precision), -d, d))
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.out_features, dtype=self.precision), -k, k))
            if self.bias:
                self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        else:  # pma
            self.Param_seeds = nn.Parameter(torch.zeros(self.num_seeds, self.in_features, dtype=self.precision).normal_(std=0.02))
            self.mha = nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_attention_heads,
                                             batch_first=True).to(self.precision)
            k = math.sqrt(1.0 / (self.num_seeds * self.in_features))
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_seeds * self.in_features, self.out_features, dtype=self.precision), -k, k))
            if self.bias:
                self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        # the head axis is pooled away, so the output is channel-free
        self.out_channels = 1

    def forward(self, node_representation: torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer
        param: node_representation: torch.Tensor -> the input tensor
        param: pos: int or list -> per-graph index or batched positions
        """
        # graph-level tensors carry a leading batch dimension in the batched
        # ShareGNN forward, (B, H, F), where the per-graph forward has (H, 1, F)
        batched = is_batched_pos(kwargs.get('pos', None))
        if node_representation.dim() != 3:
            raise ValueError(f"AttentionReadoutLayer expects a 3-D (H, 1, F) or (B, H, F) input from an "
                             f"unflattened aggregation, got {node_representation.dim()}-D. "
                             f"Set 'flatten: False' on the preceding invariant_based_aggregation layer.")
        if not batched:
            # per-graph (H, 1, F) -> (1, H, F)
            node_representation = node_representation.permute(1, 0, 2)
        x = node_representation
        if self.head_embeddings:
            x = x + self.Param_E
        if self.variant == 'transformer':
            x = self.encoder(x)
        if self.variant in ('gated', 'transformer'):
            scores = (torch.tanh(x @ self.Param_V) * torch.sigmoid(x @ self.Param_U)) @ self.Param_w
            alpha = torch.softmax(scores, dim=1)
            out = torch.einsum('bh,bhf->bf', alpha, x) @ self.Param_W
        else:  # pma
            query = self.Param_seeds.unsqueeze(0).expand(x.shape[0], -1, -1)
            attended, _ = self.mha(query, x, x, need_weights=False)
            out = attended.reshape(x.shape[0], -1) @ self.Param_W
        if self.bias:
            out = out + self.Param_b
        return self.activation(out)
