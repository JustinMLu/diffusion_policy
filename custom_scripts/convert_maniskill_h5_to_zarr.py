"""
Convert ManiSkill trajectory h5 to Diffusion Policy zarr ReplayBuffer format.

Usage:
    python custom_scripts/convert_maniskill_h5_to_zarr.py \
        --input /path/to/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
        --output data/reach_goal_heavy.zarr
"""
import argparse
import h5py
import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ManiSkill h5 trajectory file")
    parser.add_argument("--output", required=True, help="Output zarr path")
    parser.add_argument("--num-traj", type=int, default=None, help="Limit number of trajectories")
    args = parser.parse_args()

    f = h5py.File(args.input, "r")
    keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
    if args.num_traj is not None:
        keys = keys[:args.num_traj]
    print(f"Loading {len(keys)} episodes from {args.input}")

    all_obs = []
    all_actions = []
    episode_ends = []
    total_steps = 0

    for k in keys:
        ep = f[k]
        obs = ep["obs"][:]      # (L+1, obs_dim)
        actions = ep["actions"][:]  # (L, act_dim)
        L = actions.shape[0]
        obs = obs[:L]  # drop terminal obs to align with actions

        all_obs.append(obs)
        all_actions.append(actions)
        total_steps += L
        episode_ends.append(total_steps)

    f.close()

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    print(f"Total steps: {total_steps}")
    print(f"  state: {all_obs.shape} {all_obs.dtype}")
    print(f"  action: {all_actions.shape} {all_actions.dtype}")
    print(f"  episodes: {len(episode_ends)}")

    store = zarr.DirectoryStore(args.output)
    root = zarr.group(store, overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset("state", data=all_obs, chunks=(1000,) + all_obs.shape[1:])
    data.create_dataset("action", data=all_actions, chunks=(1000,) + all_actions.shape[1:])
    meta.create_dataset("episode_ends", data=episode_ends)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
