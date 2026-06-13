import torch
import torch.nn as nn
from pathlib import Path

from tools.stoppolicy.model import StopPolicyHead, FEAT_DIM, HIDDEN_DIM


class StepWrapper(nn.Module):
  """Wraps StopPolicyHead.step for ONNX export.

  Takes (feat, speed, hidden_in) and returns tuple:
    (stop_logit, dist_to_stop, desired_speed, hidden_out)
  in that exact order for ONNX serialization.
  """

  def __init__(self, policy: StopPolicyHead):
    super().__init__()
    self.policy = policy

  def forward(self, feat, speed, hidden_in):
    out_dict, hidden_out = self.policy.step(feat, speed, hidden_in)
    return (
        out_dict["stop_logit"],
        out_dict["dist_to_stop"],
        out_dict["desired_speed"],
        hidden_out,
    )


def export_onnx(model: StopPolicyHead, path):
  """Export StopPolicyHead to ONNX format (single-step streaming).

  Args:
    model: StopPolicyHead instance
    path: Path to save ONNX file (str or Path)
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  wrapped = StepWrapper(model)
  wrapped.eval()

  # Example inputs: batch_size=1, feat=(1, FEAT_DIM), speed=(1, 1), hidden=(1, HIDDEN_DIM)
  example_feat = torch.randn(1, FEAT_DIM)
  example_speed = torch.randn(1, 1)
  example_hidden = torch.zeros(1, HIDDEN_DIM)

  input_names = ["feat", "speed", "hidden_in"]
  output_names = ["stop_logit", "dist_to_stop", "desired_speed", "hidden_out"]

  dynamic_axes = {
      "feat": {0: "batch"},
      "speed": {0: "batch"},
      "hidden_in": {0: "batch"},
      "stop_logit": {0: "batch"},
      "dist_to_stop": {0: "batch"},
      "desired_speed": {0: "batch"},
      "hidden_out": {0: "batch"},
  }

  try:
    torch.onnx.export(
        wrapped,
        (example_feat, example_speed, example_hidden),
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Force legacy tracer for stability (torch 2.9 may default to dynamo)
    )
  except TypeError:
    # Older torch versions may not support dynamo param
    torch.onnx.export(
        wrapped,
        (example_feat, example_speed, example_hidden),
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )


def load_policy(checkpoint_path) -> StopPolicyHead:
  """Load a trained StopPolicyHead from a checkpoint.

  Args:
    checkpoint_path: Path to best.pt (dict with 'policy' key)

  Returns:
    StopPolicyHead instance in eval mode
  """
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  policy = StopPolicyHead()
  policy.load_state_dict(checkpoint["policy"])
  policy.eval()
  return policy


def main():
  import argparse

  parser = argparse.ArgumentParser(description="Export StopPolicyHead to ONNX")
  parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt)")
  parser.add_argument("--out", required=True, help="Output ONNX path")
  args = parser.parse_args()

  policy = load_policy(args.checkpoint)
  export_onnx(policy, args.out)
  print(f"Exported to {args.out}")


if __name__ == "__main__":
  main()
