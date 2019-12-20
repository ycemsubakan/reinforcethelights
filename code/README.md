# Simple Implementations of basic reinforcement learning algorithms. 

* Requirements/installation: You need to clone this repo `https://github.com/lcswillems/rl-starter-files` to your home, install the packages `torch-ac`, and `gym-minigrid`. (These repos also require open ai gym) . I use visdom for visualizations, but you don't have to use it. If you can't setup visdom properly, you can use some other visualization method of your choice. 
* For now, I only implemented reinforce (`--algo reinforce`), reinforce with a baseline (`--algo reinforce_wbase`), and one step actor critic (doesn't really work well for now, `--algo a2c`)
* You might be asking yourself, why shouldn't I be using rl-starter-files repo. You could. It's just that in this repo I am trying to simplify things even further, which helps to understand in my opinion. 
* Overview of structure of the code: `toyscript_v2.py` is the main file. `routines.py` is the file in which I have the classes, and the functions. The class `BaseAlgo` is used to generate episodes. Each algorithm class inherits this base class, and implements its own parameter update function. The policy model is implemented in the class `GModel`.
* An example line for running a small gridworld experiment: 
```python toyscript_v2.py --algo reinforce --env MiniGrid-Empty-5x5-v0 --save_model 1 --lr 0.0003 --num_episodes 5000 --optimizer Adam```

* Replaying a trained policy: You can use `replay_results.py` for this. At the moment I hard coded the file, but basically need to have the `modelpath` variable to point to the model which will be replayed. 


 
