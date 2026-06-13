Policy adapted from Shuijing725/CrowdNav_HEIGHT
(commit 65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05).
Source: https://github.com/Shuijing725/CrowdNav_HEIGHT
Paper: "HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation
in Crowded and Constrained Environments", Liu et al., IEEE T-ASE 2026. arXiv:2411.12150.
License: MIT (see LICENSE).

## Fork patches

- `training/networks/utils.py`: made `from training.networks.envs import VecNormalize`
  lazy (moved inside `get_vec_normalize`). Upstream imports it at module level, and
  `training/networks/envs.py` hard-imports `from baselines import bench` (and other
  `baselines.*` modules), which pulls OpenAI Baselines and TensorFlow. `utils.py` is
  transitively imported by every network module via `from training.networks.utils import
  init`, so the unpatched top-level import would force baselines + TF at inference even
  though `get_vec_normalize` is never called on the inference path. This mirrors the
  attngraph fork's identical patch.

- Stripped training-only / sim-only modules not on the inference import chain:
  `training/algo/`, `training/evaluation.py`, `training/networks/storage.py`,
  `crowd_sim/envs/*`, `crowd_sim/pybullet/` media,
  `crowd_nav/policy/{orca.py, dwa.py, social_force.py, policy_factory.py, srnn.py}`
  (pull rvo2 / training deps), `train.py`, `plot.py`, `check_env.py`, `test*.py`,
  `real_world_instruction.txt`, `figures/`. The retained tree is only what
  `from crowd_nav.configs.config import Config` and
  `from training.networks.model import Policy` import. `training/networks/envs.py` and
  `training/networks/shmem_vec_env.py` are retained but no longer imported at load: the
  `utils.py` patch makes `envs.py` lazy (only `get_vec_normalize`, never called at
  inference, pulls it), and `envs.py` references `shmem_vec_env`.

- `crowd_nav/policy/policy_factory.py` is NOT vendored on the inference path: the bridge
  builds `training.networks.model.Policy` directly, bypassing the `policy_factory -> orca
  -> rvo2` chain. (policy_factory may be retained for reference but must not be imported.)
