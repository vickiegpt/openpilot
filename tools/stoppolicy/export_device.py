"""Export the combined on-device model: image + speed + hidden -> stop decision.

The DINOv2 backbone (frozen) and the trained GRU head become one ONNX graph in
single-step streaming form, matching what stopmodeld runs each frame. Input image
is already preprocessed to (1,3,252,448) float32 in [0,1]; ImageNet normalization
happens inside the graph so the daemon only has to resize+scale.
"""
import torch
import torch.nn as nn

from tools.stoppolicy.model import StopPolicyHead, HIDDEN_DIM

IMG_H, IMG_W = 252, 448
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CombinedStopModel(nn.Module):
  def __init__(self, backbone, head):
    super().__init__()
    self.backbone = backbone
    self.head = head
    self.register_buffer("mean", _MEAN)
    self.register_buffer("std", _STD)

  def forward(self, image, speed, hidden_in):
    x = (image - self.mean) / self.std
    feats = self.backbone.forward_features(x)
    cls_tok = feats["x_norm_clstoken"]          # (1, 384)
    patch_mean = feats["x_norm_patchtokens"].mean(dim=1)  # (1, 384)
    feat = torch.cat([cls_tok, patch_mean], dim=1)        # (1, 768)
    out, hidden_out = self.head.step(feat, speed, hidden_in)
    return out["stop_logit"], out["dist_to_stop"], out["desired_speed"], hidden_out


def build_combined(policy_state, device="cpu"):
  backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
  head = StopPolicyHead()
  if policy_state is not None:
    head.load_state_dict(policy_state)
  m = CombinedStopModel(backbone, head).to(device).eval()
  for p in m.parameters():
    p.requires_grad_(False)
  return m


def export_combined(model, path):
  model.eval()
  img = torch.zeros(1, 3, IMG_H, IMG_W)
  speed = torch.zeros(1, 1)
  hidden = torch.zeros(1, HIDDEN_DIM)
  torch.onnx.export(
    model, (img, speed, hidden), str(path),
    input_names=["image", "speed", "hidden_in"],
    output_names=["stop_logit", "dist_to_stop", "desired_speed", "hidden_out"],
    opset_version=17, dynamo=False,
  )


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["policy"]
  m = build_combined(policy_state=state, device="cpu")
  export_combined(m, args.out)
  print(f"exported combined device model -> {args.out}")


if __name__ == "__main__":
  main()
