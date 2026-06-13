import torch

from tools.stoppolicy.model import StopPolicyHead, AuxPresenceHead, FEAT_DIM, HIDDEN_DIM


def test_policy_step_shapes():
  m = StopPolicyHead()
  feat = torch.randn(8, FEAT_DIM)
  speed = torch.randn(8, 1)
  h = m.initial_hidden(8)
  out, h2 = m.step(feat, speed, h)
  assert out["stop_logit"].shape == (8, 1)
  assert out["dist_to_stop"].shape == (8, 1)
  assert out["desired_speed"].shape == (8, 1)
  assert h2.shape == (8, HIDDEN_DIM)


def test_policy_sequence_matches_steps():
  m = StopPolicyHead().eval()
  feats = torch.randn(2, 5, FEAT_DIM)
  speeds = torch.randn(2, 5, 1)
  with torch.no_grad():
    seq_out = m.forward_sequence(feats, speeds)
    h = m.initial_hidden(2)
    for t in range(5):
      step_out, h = m.step(feats[:, t], speeds[:, t], h)
    assert torch.allclose(seq_out["stop_logit"][:, -1], step_out["stop_logit"], atol=1e-5)


def test_aux_head_shapes():
  a = AuxPresenceHead()
  logits = a(torch.randn(4, FEAT_DIM))
  assert logits.shape == (4, 2)  # has_light, has_sign
