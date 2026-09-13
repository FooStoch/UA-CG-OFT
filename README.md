# UA-CG-OFT: Uncertainty-Aware and Counterfactual-Grounding Optimized Fine-Tuning for Vision-Language-Action Models

**Research paper**: a 

**Project Github**: https://github.com/FooStoch/UA-CG-OFT

**Summary video**: a

**Full project zip**: a

UA-CG-OFT is a fine-tuning framework for Vision-Language-Action (VLA) models. It extends the OpenVLA-OFT training stack with two auxiliary objectives:

- **Adaptive uncertainty:** uses action risk from action-token representations and action statistics to predict an uncertainty score for adaptive flow matching reasoning
- **Region alignment:** aligns language-conditioned visual patch features with target regions using negative-region contrastive loss and counterfactual action supervision

We retain the OpenVLA-compatible continuous action heads (L1 regression, action chunking, etc.), LoRA fine-tuning, and LIBERO/ALOHA workflows. The UA-CG-OFT additions are opt-in, so ordinary OpenVLA fine-tuning remains available.

## Repository layout

- `vla-scripts/finetune.py` — main fine-tuning entry point and UA-CG-OFT configuration.
- `prismatic/models/ua_cg_oft.py` — adaptive-uncertainty and region-alignment module.
- `experiments/robot/libero/run_libero_eval.py` — LIBERO evaluation entry point.
- `experiments/robot/openvla_utils.py` and `experiments/robot/robot_utils.py` — model loading and action-generation utilities.
- `LIBERO.md` and `ALOHA.md` — benchmark-specific setup and commands.

## System Requirements

Inference:

- 1 GPU with approximately 16 GB VRAM for LIBERO simulation benchmark tasks.
- 1 GPU with approximately 18 GB VRAM for ALOHA robot tasks.

Training:

- Between 1 and 8 GPUs with approximately 27–80 GB VRAM each, depending on the training setup and batch size, using the default `bfloat16` dtype.
- UA-CG-OFT adds the uncertainty and region-alignment heads; actual memory use also depends on whether region masks and counterfactual examples are included in a batch.

## Quick Start

First, set up a conda environment (see [SETUP.md](SETUP.md)). Then run the Python example below to load a compatible VLA checkpoint and generate an action chunk from the included LIBERO observation.

```python
import pickle

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# See GenerateConfig in experiments/robot/libero/run_libero_eval.py for all options.
cfg = GenerateConfig(
    pretrained_checkpoint="/PATH/TO/COMPATIBLE/CHECKPOINT",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="libero_spatial_no_noops",
)

# Load the VLA policy, processor, continuous action head, and proprio projector.
vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# The observation contains full_image, wrist_image, state, and task_description.
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

actions = get_vla_action(
    cfg,
    vla,
    processor,
    observation,
    observation["task_description"],
    action_head,
    proprio_projector,
)
print("Generated action chunk:")
for action in actions:
    print(action)
```

For a UA-CG-OFT-trained checkpoint, load its separate `ua_cg_module--<step>_checkpoint.pt` with `get_ua_cg_module` from `experiments.robot.openvla_utils`, then pass it as `ua_cg_module=` to `get_vla_action` (or `get_action`). This auxiliary module is used by the adaptive flow-matching path; the baseline example above works without it.

## Installation

See [SETUP.md](SETUP.md) for environment setup. For LIBERO, clone and install the benchmark and its additional requirements:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## UA-CG-OFT Fine-Tuning

Use [vla-scripts/finetune.py](vla-scripts/finetune.py) as the training entry point. Add the following flag to enable UA-CG-OFT:

```bash
--use_ua_cg_oft True
```

The available UA-CG-OFT controls are `ua_cg_hidden_dim`, `ua_uncertainty_loss_weight`, `ua_uncertainty_error_scale`, `cg_grounding_loss_weight`, `cg_grounding_dice_weight`, `cg_contrastive_loss_weight`, `cg_equivariance_loss_weight`, and `cg_contrastive_temperature`.

The uncertainty loss is active whenever predicted and target actions are available. Region alignment additionally uses `target_masks` and can use `negative_target_masks`; counterfactual action consistency uses the optional existing `counterfactual_*` batch fields. Missing optional region/counterfactual fields cause only their associated loss terms to be skipped.

Each enabled UA-CG-OFT run saves the auxiliary state separately as `ua_cg_module--<step>_checkpoint.pt`. Keep it with the corresponding model, action-head, and projector checkpoints.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning and evaluating in the LIBERO simulation benchmark. See [ALOHA.md](ALOHA.md) for the real-world ALOHA workflow.

When evaluating a checkpoint, match the action-head mode, image count, proprioception setting, LoRA rank, and crop setting used for training.

## Support

If you run into any issues, please email Victor Young (victoryfoo27@gmail.com) to bring the issue to his attention.
