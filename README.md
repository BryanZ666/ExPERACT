# ExPERACT
Explainable PERACT for robotic manipulation

![](./viz/episode_close_jar_eval_7.gif)
![](./viz/episode_light_bulb_in_eval_4.gif)
![](./viz/episode_meat_off_grill_eval_1.gif)
![](./viz/episode_open_drawer_eval_1.gif)

![](./viz/episode_push_buttons_eval_13.gif)
![](./viz/episode_slide_block_to_color_target_eval_8.gif)
![](./viz/episode_stack_blocks_eval_24.gif)
![](./viz/episode_turn_tap_eval_25.gif)

The demonstration data is downloaded from Project PreAct. Train (100 episodes), validation (25 episodes), and test (25 episodes)

--  https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ

```
export COPPELIASIM_ROOT_DIR="$HOME/Data/python/RLBench/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04"

export COPPELIASIM_ROOT="$HOME/Data/python/RLBench/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04"

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH:$COPPELIASIM_ROOT_DIR

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT_DIR

export HYDRA_FULL_ERROR=1

cd /directory/to/experact/

python main.py
```