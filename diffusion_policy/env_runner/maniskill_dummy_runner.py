from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class ManiSkillDummyRunner(BaseLowdimRunner):
    """No-op runner for ManiSkill tasks where the eval env can't run in this conda env.
    Evaluate checkpoints separately using the maniskill conda environment."""

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return {'test/mean_score': 0.0}
