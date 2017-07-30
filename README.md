# A way to visualize important objects in Reinforcement Learning Scene

### Description
This is not an implementation of a paper, but an interesting idea that I am trying. <br />
Train multi head attention model that focus on important objects. From extracted objects model does prediction given state to action, where labels are given from pretrained model. By embedding attention into model we get nice visualization of important objects in the scene. <br />
This model use cnn to extract object features, concatenated with normalized positions x, y. After that there are N (2 for Pong) heads that focus on different objects (multi-head attention). Producing weights for visualization. And outputting N objects that are passed to FC, to make prediction from state to action labeled by pretrained model. <br />
Since state is 4 consecutive images, there are averaged in visualization.

### Results
Folder "results" contain 100 random samples, game Pong. <br /> <br />
Example 1 <br />
![Result 1](results/attention9.png?raw=true "multi head attention")

---------
---------

Example 2 <br />
![Result 2](results/attention10.png?raw=true "multi head attention")

### Dependencies
OpenAi Baselines pretrained models
```sh
$ pip3 install baselines
```
Download trained Pong model
```sh
$ python3.5 -m baselines.deepq.experiments.atari.download_model --blob model-atari-duel-pong-1 --model-dir ./model-atari-duel-pong-1
```

### Run
Not necessary, already trained attention model in "model-torch-2heads" folder
```sh
$ python3.5 train_attn.py
```
Not necessary, results of visualization are in "results" folder
```sh
$ python3.5 visualize.py
```

### Files
"model_attn_double.py" - attention model <br />
"train_attn.py" - trains attention model <br />
"visualize.py" - produces visualization from trained attention model <br />
"visualize_model.py" - used for saving images <br />

### Future work
Other environments and variations of models will be tried. Proof of concept is ready in "results" folder