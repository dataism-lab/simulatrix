import os

import torch
from ray.rllib.agents.ppo import PPOTrainer


class CustomPPOTrainer(PPOTrainer):
    """
    Saving torch model, not pkl
    """

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)

        model = self.get_policy().model
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, "checkpoint_state_dict.pth"))

        return checkpoint_path
