# grab all their training data

link to their goog drive is: wget https://diffusion-policy.cs.columbia.edu/data/training/

```bash
# Here are all of them
wget https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht_real.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
```


# custom script #1: convert maniskill h5 to zarr

```bash

# convert maniskill .h5 (with obs) to zarr
python custom_scripts/convert_maniskill_h5_to_zarr.py \
    --input /path/to/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
    --output data/reach_goal_heavy.zarr \
    --num-traj <num desired trajectories>

# can drop --num-traj argument to convert everything if desired
```