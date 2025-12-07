import torch

class Analyzer:
    def __init__(self, model):
        self.model = model
        self.cross_attention_weights = []
        self.hooks = []
        self._register_hooks()

    # https://docs.pytorch.org/docs/stable/notes/modules.html#module-hooks
    def _hook_fn(self, layer_idx, module, inputs, output):
        x = inputs[0]
        with torch.no_grad():
            _, attention = module.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )

        if len(self.cross_attention_weights) <= layer_idx:
            self.cross_attention_weights.extend(
                [None] * (layer_idx + 1 - len(self.cross_attention_weights))
            )
        self.cross_attention_weights[layer_idx] = attention.detach().cpu()

    def _register_hooks(self):
        for i, layer in enumerate(self.model.cross_asset_encoder.layers):
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
            hook = layer.register_forward_hook(
                lambda module, inputs, output, layer_idx=i: self._hook_fn(layer_idx, module, inputs, output)
            )
            self.hooks.append(hook)

    def _extract_real_attention(self, x):
        self.cross_attention_weights = []
        print(f"Input shape: {x.shape}")
        with torch.no_grad():
            self.model(x)
        return self.cross_attention_weights

    def _last_layer_attention(self, x):
        attention_weights = self._extract_real_attention(x)
        attention = attention_weights[-1] 
        return attention.mean(dim=0)

    def averaged_attention(self, x):
        attention = self._last_layer_attention(x)
        return attention.mean(dim=0).numpy()

    def headwise_attention(self, x):
        attention = self._last_layer_attention(x)
        return attention.numpy()

