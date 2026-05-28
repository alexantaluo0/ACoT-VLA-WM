"""Policy transforms for the Go2 robot."""

import dataclasses
from typing import ClassVar
from collections.abc import Sequence
import numpy as np
import torch
import copy

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class Go2Inputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }

    def __call__(self, data: dict) -> dict:

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = copy.deepcopy(state)
        # state[14:]  = state[14:] * 120
        if self.state_mask is not None:
            state[self.state_mask] = 0

        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}


        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        # Add actions if present
        if "actions" in data:
            actions = data["actions"]
            if self.action_mask is not None:
                actions[:, self.action_mask[:actions.shape[1]]] = 0
            actions = transforms.pad_to_dim(actions, self.action_dim)



            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go2Outputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])} 


@dataclasses.dataclass(frozen=True)
class Go2ACOTInputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    prompt_map_inject_to_training: dict[str, str] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    acot_action_generation: Sequence[Sequence[int]] | None = None

    # 将state和action重新排列为24-dim布局，匹配40-dim AgiBot actions的布局
    def slice_state_and_action(self, data):
        # Slice state to 24-dim layout matching action reorder for 40-dim AgiBot actions:
        # arm joints (14), grippers (2), head (3), waist (5).
        state_indices = None
        if len(data["state"]) == 183:
            state_indices = (
                list(range(54, 68)) + [0, 1] + list(range(96, 99)) + list(range(99, 104))
            )

        if len(data["state"]) == 159:
            state_indices = (
                list(range(30, 44)) + [0, 1] + list(range(72, 75)) + list(range(75, 80))
            )
        if state_indices is not None:
            data["state"] = data["state"][state_indices]

            # # 写数据，确认载入state数据是否正确
            # with open("/tmp/go2_state_debug.log", "a") as f:
            #     f.write(f"{data['state']}\n")

            if len(data["state"]) != 24:
                raise ValueError(
                    f"Go2ACOTInputs: expected 24-dim state after slicing, got {len(data['state'])}."
                )

        if "actions" in data:
            n_a = data["actions"].shape[1]
            if n_a == 40:
                # AgiBot flat action -> train order:
                # arms(14), grippers(2), head(3), waist(5).
                data["actions"] = np.column_stack(
                    (
                        data["actions"][:, 16:30],
                        data["actions"][:, 0:2],
                        data["actions"][:, 30:33],
                        data["actions"][:, 33:38],
                    )
                )
            elif n_a in (21, 24):
                # Already ordered (e.g. 21-dim legacy stack, or 24-dim full-body joint order).
                pass
            else:
                raise ValueError(f"Go2ACOTInputs: unsupported action width {n_a}, expected 40, 24, or 21.")
        return data
    
    def random_inject_prompt(self, data):
        color_episode_pairs_for_task_sort_packages = {
            'white': [
                0, 9, 11, 15, 18, 19, 22, 34, 39, 41, 49, 52, 55, 62, 66, 68, 69, 73, 74, 81, 90, 96, 111, 120, 123, 125, \
                126, 129, 137, 139, 145, 149, 156, 157, 158, 164, 166, 170, 185, 187, 188, 190, 202, 207, 208, 209, 211, \
                213, 218, 220, 221, 226, 227, 228, 229, 230, 236, 246, 250, 251, 252, 260, 264, 266, 274, 275, 279, 282, 283
            ],
            'red': [
                1, 3, 5, 12, 13, 23, 24, 25, 26, 27, 28, 33, 38, 51, 53, 57, 58, 59, 61, 64, 76, 77, 80, 82, 84, 87, 91, 100, 102, 103, 105, \
                110, 114, 116, 118, 128, 142, 143, 148, 150, 152, 162, 163, 165, 167, 168, 173, 174, 176, 179, 186, 191, 192, 194, 197, 199, \
                205, 206, 214, 217, 222, 224, 234, 237, 238, 240, 241, 242, 243, 244, 247, 248, 253, 257, 262, 268, 270, 272, 273, 276, 277, 278, 284
            ],
            'black': [
                2, 14, 16, 17, 20, 29, 30, 31, 32, 37, 40, 42, 43, 44, 45, 46, 47, 48, 50, 56, 60, 65, 67, 70, 71, 72, 75, 78, 79, 83, 85, 88, \
                92, 95, 97, 99, 104, 107, 109, 112, 115, 117, 119, 122, 124, 127, 131, 132, 140, 141, 146, 147, 151, 155, 159, 160, 171, 172, 175, \
                177, 178, 180, 183, 184, 200, 203, 210, 212, 225, 233, 235, 245, 254, 255, 256, 259, 261, 263, 267, 269, 271, 280
            ],
            'yellow': [
                4, 6, 7, 8, 10, 21, 35, 36, 54, 63, 86, 89, 93, 94, 98, 101, 106, 108, 113, 121, 130, 133, 134, 135, 136, 138, 144, 153, \
                154, 161, 169, 181, 182, 189, 193, 195, 196, 198, 201, 204, 215, 216, 219, 223, 231, 232, 239, 249, 258, 265, 281, 285
            ]
        }

        task_name = data["task"]
        episode_idx = data["episode_index"]
        if self.prompt_map_inject_to_training is not None and task_name in self.prompt_map_inject_to_training:
            default_prompt = self.prompt_map_inject_to_training[task_name][0]
            inject_prob = self.prompt_map_inject_to_training[task_name][1]

            if task_name == "Sort packages":
                for key, value in color_episode_pairs_for_task_sort_packages.items():
                    if episode_idx in value:
                        default_prompt = default_prompt.replace("<color>", key)
                        break

            if np.random.rand() < inject_prob:
                data["prompt"] = default_prompt
    
        return data

    def __call__(self, data: dict) -> dict:
        data = self.slice_state_and_action(data)
        state = copy.deepcopy(transforms.pad_to_dim(data["state"], self.action_dim))
        if self.state_mask is not None:
            state[np.array(self.state_mask)] = 0

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}

        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if self.acot_action_generation is not None and "actions" in data:
            action_horizons = self.acot_action_generation[0]
            joint_action_shifts = self.acot_action_generation[1]

            raw_data = data["actions"]
            keys = ["coarse_actions", "actions"]
            for idx, key in enumerate(keys):
                action_horizon = action_horizons[idx]
                joint_action_shift = joint_action_shifts[idx]
                required_length = (action_horizon - 1) * joint_action_shift + 1
                data[key] = copy.deepcopy(raw_data[:required_length:joint_action_shift])
                assert len(data[key]) == action_horizon
        for key in ['coarse_actions', 'actions']:
            if key in data:
                if self.action_mask is not None:
                    data[key][:, np.array(self.action_mask)[:data[key].shape[1]]] = 0
                data[key] = transforms.pad_to_dim(data[key], self.action_dim)
                inputs[key] = data[key]

        if "task" in data: # training
            data = self.random_inject_prompt(data)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go2ACOTOutputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    # Truncate model outputs (padded to action_dim) to the robot command dimension.
    robot_action_dim: int = 21

    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        return {key: np.asarray(data[key][:, : self.robot_action_dim]) for key in keys if key in data}