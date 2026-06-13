import torch
import torch.nn as nn

FEAT_DIM = 768
HIDDEN_DIM = 256


class StopPolicyHead(nn.Module):
  """Step-wise GRU policy head. Explicit hidden state -> exports to ONNX as a
  single-step model and streams on-device; forward_sequence is the training path."""

  def __init__(self):
    super().__init__()
    self.inp = nn.Sequential(nn.Linear(FEAT_DIM + 1, 512), nn.ReLU(), nn.Linear(512, 256))
    self.cell = nn.GRUCell(256, HIDDEN_DIM)
    self.stop_head = nn.Linear(HIDDEN_DIM, 1)
    self.dist_head = nn.Linear(HIDDEN_DIM, 1)
    self.speed_head = nn.Linear(HIDDEN_DIM, 1)

  def initial_hidden(self, batch):
    return torch.zeros(batch, HIDDEN_DIM, device=next(self.parameters()).device)

  def step(self, feat, speed, hidden):
    x = self.inp(torch.cat([feat, speed], dim=-1))
    h = self.cell(x, hidden)
    out = {"stop_logit": self.stop_head(h),
           "dist_to_stop": torch.nn.functional.softplus(self.dist_head(h)),
           "desired_speed": torch.nn.functional.softplus(self.speed_head(h))}
    return out, h

  def forward_sequence(self, feats, speeds):
    B, T, _ = feats.shape
    h = self.initial_hidden(B)
    outs = {"stop_logit": [], "dist_to_stop": [], "desired_speed": []}
    for t in range(T):
      out, h = self.step(feats[:, t], speeds[:, t], h)
      for k in outs:
        outs[k].append(out[k])
    return {k: torch.stack(v, dim=1) for k, v in outs.items()}


class AuxPresenceHead(nn.Module):
  """has_light / has_sign presence from a single frame's features (aux loss)."""

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 2))

  def forward(self, feat):
    return self.net(feat)
