import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import re


class ResNetBBoxModel(nn.Module):
    def __init__(self, resnet_version="resnet50", num_classes=265,
                 bbox_heads=[True, True, True, True], class_head=True,
                 input_resolution=(224, 224), per_class_bbox: bool = False):
        super().__init__()

        self.per_class_bbox = per_class_bbox

        self.input_resolution = input_resolution  # Fixed input resolution

        # Load the chosen resnet variant (pretrained on ImageNet)
        backbone = getattr(models, resnet_version)(pretrained=True)

        # Truncate only the final FC layer, keep avgpool
        truncated = list(backbone.named_children())[:-1]  # keep avgpool, remove fc
        truncated_odict = OrderedDict(truncated)
        self.backbone = nn.Sequential(truncated_odict)

        # Flatten the avgpool output
        self.flatten = nn.Flatten(start_dim=1)

        # Infer output size after avgpool (e.g. 2048 for resnet50)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *self.input_resolution)
            dummy_output = self.backbone(dummy_input)  # [1, 2048, 1, 1]
            dummy_output = self.flatten(dummy_output)  # [1, 2048]
            self.backbone_output_dim = dummy_output.shape[1]

        # Heads directly attached after flattening avgpool
        self.class_head = (
            nn.Linear(self.backbone_output_dim, num_classes)
            if class_head else nn.Identity()
        )

        self.bbox_heads = nn.ModuleList([
            (nn.Linear(self.backbone_output_dim, num_classes) if per_class_bbox
             else nn.Linear(self.backbone_output_dim, 1)) if active else nn.Identity()
            for active in bbox_heads
        ])

        # Initialize the custom layers
        self._initialize_custom_weights()

        # Print info
        print(f"Model loaded with {resnet_version} backbone.\nBackbone layers are named:")
        for name, child in self.backbone.named_children():
            print(name)

    def _initialize_custom_weights(self):
        layers_to_init = []

        if not isinstance(self.class_head, nn.Identity):
            layers_to_init.append(self.class_head)

        for head in self.bbox_heads:
            if not isinstance(head, nn.Identity):
                layers_to_init.append(head)

        for layer in layers_to_init:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        """ Freezes all backbone layers. """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """ Unfreezes all backbone layers. """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone_until(self, freeze_layer="layer2"):
        """
        Freeze all backbone layers up to (and including) the specified layer.
        Unfreeze all subsequent layers.

        Typical valid values for freeze_layer (for ResNet) are:
            "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"
        If freeze_layer is None or an empty string, no layers are frozen.
        """
        # Build a name->index map for clarity.
        index_map = {}
        for idx, (name, _) in enumerate(self.backbone.named_children()):
            index_map[name] = idx

        if not freeze_layer:
            return  # No freeze

        target_idx = None
        for name, idx in index_map.items():
            if freeze_layer in name:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Requested freeze layer '{freeze_layer}' not found in backbone modules.")

        for idx, (name, child) in enumerate(self.backbone.named_children()):
            if idx <= target_idx:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        assert x.shape[-2:] == self.input_resolution, \
            f"Input image must have resolution {self.input_resolution}, but got {x.shape[-2:]}"

        # Pass through truncated backbone
        x = self.backbone(x)  # [B, 2048, 1, 1]
        x = self.flatten(x)  # [B, 2048]

        # Classification head: returns Tensor or Identity => T or NxN
        class_out = self.class_head(x)
        if isinstance(self.class_head, nn.Identity):
            class_out = None  # We'll interpret Identity as "no classification"

        # BBox heads: if a head is Identity, interpret as "no output" => None
        bbox_outs = []
        per_class = getattr(self, "per_class_bbox", False)  # default False for old checkpoints
        for head in self.bbox_heads:
            if isinstance(head, nn.Identity):
                bbox_outs.append(None)
            else:
                y = head(x)  # shape [B, 1] or [B, num_classes]
                bbox_outs.append(y)

        return class_out, bbox_outs

    def _get_module(self, module_name: str) -> nn.Module:
        """
        Resolve a dotted module_name (e.g. "backbone.layer3.4.bn2")
        into the actual nn.Module object inside self.
        """
        parts = module_name.split('.')
        module: nn.Module = self
        for p in parts:
            # 1) direct attribute (e.g. self.backbone)
            if hasattr(module, p):
                module = getattr(module, p)
            # 2) named submodule (Sequential._modules, ModuleList, etc.)
            elif p in module._modules:
                module = module._modules[p]
            # 3) integer index into list/Sequential
            else:
                try:
                    idx = int(p)
                    module = module[idx]
                except Exception:
                    raise ValueError(f"Cannot resolve '{p}' in '{module_name}'")
        return module

    def forward_features(self,
                         x: torch.Tensor,
                         module_name: str = "backbone",
                         include_predictions: bool = False,
                         flatten: bool = True
                         ):
        """
        Run a forward pass and return the activation of the named module.

        If include_predictions=True, also returns (class_out, bbox_outs)
        exactly as produced by `self.forward(x)`.

        Supports:
          - stage/block hooks like "backbone.layer3" or "backbone.layer3.4"
            (block hooks include their internal skip+final ReLU)
          - leaf hooks like "backbone.layer3.4.bn2"
          - disambiguated ReLUs via ".reluX" where X is any positive integer
        """
        # 1) check input size
        assert x.shape[-2:] == self.input_resolution, (
            f"Expected input {self.input_resolution}, got {x.shape[-2:]}"
        )

        # 2) handle reluX disambiguation
        m = re.match(r"^(.*)\.relu(\d+)$", module_name)
        if m:
            parent, num = m.group(1), int(m.group(2))
            relu_mod = self._get_module(parent + ".relu")
            activations = []
            handle = relu_mod.register_forward_hook(
                lambda m, inp, out: activations.append(out)
            )
            preds = self.forward(x)  # full forward to get predictions and trigger hooks
            handle.remove()

            idx = num - 1
            if idx < 0 or idx >= len(activations):
                raise RuntimeError(
                    f"Requested '{module_name}' but saw only {len(activations)} ReLU calls"
                )
            feat = activations[idx]

        else:
            # 3) forbid plain .relu
            if module_name.endswith(".relu"):
                raise ValueError("Use '.reluX' (e.g. '.relu1', '.relu2', …) to pick which ReLU.")

            # 4) generic module hook
            target = self._get_module(module_name)
            activations = []
            handle = target.register_forward_hook(
                lambda m, inp, out: activations.append(out)
            )
            preds = self.forward(x)
            handle.remove()

            if not activations:
                raise RuntimeError(f"No activations captured for '{module_name}'")
            feat = activations[-1]

        # 5) optional flatten (keep batch dim)
        if flatten:
            feat = feat.reshape(feat.shape[0], -1)

        # 6) return
        if include_predictions:
            return feat, preds
        else:
            return feat


# --- backward-compatibility alias ------------------------------------------
AvgPool_SDXLdataset = ResNetBBoxModel      # old code can `from LIB_model import AvgPool_SDXLdataset`
__all__ = ["ResNetBBoxModel", "AvgPool_SDXLdataset"]
