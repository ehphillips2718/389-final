**This is a modified version of FOOTSIES that supports training a Reinforcement Learning agent through the Gymnasium API.**

Information regarding this modification, including installation and usage, is present at the [end of the README](#footsies-gym).

# Footsies

FOOTSIES is a 2D fighting game where players can control character movement horizontally 
and use one attack button to perform normal and special moves to defeat their opponent.
While the controls (and graphics) are super simple, 
FOOTSIES retains the fundamental feeling of fighting game genre 
where spacing, hit confirm and whiff punish are keys to achieve victory.

<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_00.jpg">

This is a fun little project for the fighting game community and 
for own practice developing a game by myself.
I am objectively bad at art and music but that went kinda well with theme of the game. 
The animation in this game are, obviously, heavily inspired by the most iconic you-know-who fighting game character.

Although I only tested this game mostly with CPU, when I actually tried this game with my friend, 
we actually had a lot of fun! So, I hope that everyone try this game and have some fun as well :D

<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_01.jpg">

<h3>Download</h3> 

<b><u><a href="https://github.com/hifight/Footsies/releases" download>FOOTSIES</a></u></b>

※You can config keys/buttons input when the game is launched, although XInput can't be set on config windows, 
XInput controller should work fine in the game.


<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_03.jpg">


<h3>Mechanics</h3> 

- There is no health bar. The round is lost after being hit by special moves.
- There is, however, guard bar. You can block opponent attack up to three times. After that, every attack will cause guard break.
- There are two type of normal moves, neutral attack and forward/backward attack.
- There are two type of special moves which can be performed by holding and then release attack button.
One can be performed by neutral release, and forward/backward release for the other one.
- If normal moves connect with the opponent, whether on hit or block, it can be canceled into neutral special move by pressing an attack button again.
- Forward and backward dashes can be performed by pressing forward/backward twice.
- Hitbox/hurtbox/frame information can be toggle on and off by pressing F12.
- Press F1 to pause/resume the game. While pausing, pressing F2 will play the game for 1 frame.


<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_04.jpg">


Whether you like the game or not, feel free to leave some comments about your experience on my <b><u><a href="https://twitter.com/">twitter</a></u></b>

If you like the game then invite your friend to play this game too! Seeing some tournament for this game would be a dream come true for me.

## Footsies-Gym

This repository contains the source code for the modified version of FOOTSIES, as well as the Gymnasium environment.
If binary releases aren't available for your platform, the game has to be built from source (Unity Engine version 2022.3.10f1 was used).

The `footsies_gym` folder is not part of the game's project. This folder contains the Gymnasium environment which allows interaction with the game using Python.

### Installation

The environment was only tested for Python versions +3.8.10.
The only dependency is the `gymnasium` module ([installation instructions](https://github.com/Farama-Foundation/Gymnasium#installation)). This module doesn't officially support Windows, but it works in this project.

In order to use the environment, install the `footsies-gym` module at the root of the project (using a virtual environment is recommended):

```
pip install -e footsies-gym
```

### Usage

The environment can be instantiated in Python with `gymnasium.make` or by directly instantiating `footsies_gym.envs.footsies.FootsiesEnv`:

```python
env = gymnasium.make("FootsiesEnv-v0")
# or
env = FootsiesEnv(...)
```

Make sure the environment is properly terminated (`env.close()`) so that the socket and game are gracefully closed.
If a new episode has to be started with `env.reset()` before the environment has terminated/truncated, then `env.hard_reset()` should be called (which will close and re-open all resources).

### Game command-line arguments

The game binary itself accepts some command-line arguments, but they don't need to be manually specified for normal usage:

- `--training`: setup the game for training (VS CPU battle with custom training actors which serve as the players)
- `--fast-forward`: fast-forward the game 20x
- `--synced`: use synchronous socket communication
- `--mute`: mute all sound
- `--{p1, p2}-bot`: Player 1/2 is the in-game AI bot (`TrainingBattleAIActor`)
- `--{p1, p2}-player`: Player 1/2 is human-controlled (`TrainingPlayerActor`)
- `--{p1, p2}-spectator`: Player 1/2 will have a socket from which the environment state can be viewed, but actions are not specified through the socket (`TrainingActorRemoteSpectator`). This argument only makes sense in conjunction with `--{p1, p2}-bot` or `--{p1, p2}-player`
- `--{p1, p2}-address`: the address of the socket used for training
- `--{p1, p2}-port`: the port of the socket used for training
- `--{p1, p2}-no-state`: specify that no environment state is to be sent to the remote player 1/2. No effect if Player 1/2 is a spectator

If neither `--{p1, p2}-bot` nor `--{p1, p2}-player` are specified then Player 1/2 will be a remote actor (`TrainingRemoteActor`).
A socket will be associated with the actor through which actions and environment state are communicated.

