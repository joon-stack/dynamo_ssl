import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict

import wandb
from utils.video import VideoRecorder
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets
from datasets.vqbet_repro import TrajectorySlicerDataset
from models.encoder.dinosaur import Visual_Encoder


if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="eval_configs", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    encoder = hydra.utils.instantiate(cfg.encoder)
    state_dict = torch.load(encoder.ckp_path)['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else "module." + k  # 접두어 추가 또는 제거
        new_state_dict[name] = v

    msg = encoder.load_state_dict(new_state_dict, strict=False)
    print(msg)
    encoder = encoder.to(cfg.device).eval()

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    state_dict = torch.load(cbet_model.ckp_path)['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else "module." + k  # 접두어 추가 또는 제거
        new_state_dict[name] = v
    cbet_model.load_state_dict(new_state_dict)
    cbet_model = cbet_model.to(cfg.device).eval()

    env = hydra.utils.instantiate(cfg.env.gym)
    if "use_libero_goal" in cfg.data:
        with torch.no_grad():
            # calculate goal embeddings for each task
            goals_cache = []
            for i in range(10):
                idx = i * 50
                last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
                last_obs = last_obs.to(cfg.device)
                embd = encoder(last_obs)[0]  # V E
                embd = einops.rearrange(embd, "V E -> (V E)")
                goals_cache.append(embd)

        def goal_fn(goal_idx):
            return goals_cache[goal_idx]
    else:
        empty_tensor = torch.zeros(1)

        def goal_fn(goal_idx):
            return empty_tensor

    run = wandb.init(
        project=cfg.wandb.project,
        # entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @torch.no_grad()
    def eval_on_env(
        cfg,
        num_evals=cfg.num_env_evals,
        num_eval_per_goal=1,
        videorecorder=None,
        epoch=None,
    ):
        def embed(enc, obs):
            obs = (
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            )  # 1 V C H W
    
            obs = obs[0] # [V, 3, H, W]
            features = enc.vis_encoder(obs)
            result = enc(features) # [V, num_slots, slot_dim]
            # result = einops.rearrange(result, "1 V E -> (V E)")
            return result

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        env.seed(cfg.seed)
        for goal_idx in range(num_evals):
            if videorecorder is not None:
                videorecorder.init(enabled=True)
            for i in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)
                this_obs = env.reset(goal_idx=goal_idx)  # V C H W
                assert (
                    this_obs.min() >= 0 and this_obs.max() <= 1
                ), "expect 0-1 range observation"
                this_obs_enc = embed(encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                goal = goal_fn(goal_idx)  # V C H W
                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                    # goal = embed(encoder, goal)
                    goal = goal.unsqueeze(0).repeat(cfg.eval_window_size, 1)
                    action = cbet_model(obs.squeeze(1))
                    action = action[0]  # remove batch dim; always 1
                    if cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action.cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    if videorecorder.enabled:
                        videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    goal = goal_fn(goal_idx)
                avg_reward += total_reward
                if cfg.env.gym.id == "pusht":
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info["final_coverage"])
                elif cfg.env.gym.id == "blockpush":
                    avg_max_coverage.append(info["moved"])
                    avg_final_coverage.append(info["entered"])
                completion_id_list.append(info["all_completions_ids"])
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    metrics_history = []
    reward_history = []

    


    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        num_eval_per_goal=cfg.num_final_eval_per_goal,
        videorecorder=video,
        epoch=cfg.epochs,
    )
    reward_history.append(avg_reward)
    if cfg.env.gym.id in ["pusht", "blockpush"]:
        metric_final = "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        metrics = {
            f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
            f"{metric_final} max": max(final_coverage),
            f"{metric_final} min": min(final_coverage),
            f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
            f"{metric_max} max": max(max_coverage),
            f"{metric_max} min": min(max_coverage),
        }
        wandb.log(metrics)
        metrics_history.append(metrics)

    with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    if cfg.env.gym.id == "pusht":
        final_eval_on_env = max([x["final coverage mean"] for x in metrics_history])
    elif cfg.env.gym.id == "blockpush":
        final_eval_on_env = max([x["entered mean"] for x in metrics_history])
    elif cfg.env.gym.id == "libero_goal":
        final_eval_on_env = max(reward_history)
    elif cfg.env.gym.id == "kitchen-v0":
        final_eval_on_env = avg_reward
    wandb.log({"final_eval_on_env": final_eval_on_env})
    return final_eval_on_env


if __name__ == "__main__":
    main()
