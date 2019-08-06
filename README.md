# Flappy Bird (Python/Pygame)

Clone of the classic Flappy Bird built with Python and Pygame, with two modes: **manual play** (keyboard) and **NEAT self-play** where a neural network learns to play through neuroevolution.

## Context
- **Date:** 2019
- **Type:** Personal Project

## Tech Stack
- Python 3
- Pygame
- [neat-python](https://neat-python.readthedocs.io/) (for the AI mode)

## How to Run
```bash
pip install pygame neat-python
python main.py
```

From the menu:
- `1` — Manual mode
- `2` — NEAT (AI) mode
- `ESC` — Quit

### Manual mode
- `SPACE` — Jump
- `R` — Restart after losing
- `ESC` — Back to menu

### NEAT mode
A population of birds evolves across generations. Inputs to each network: bird `y` position and distance to the top/bottom of the next pipe. Output: jump if > 0.5. Press `ESC` to stop training and return to the menu.

Hyperparameters (population size, mutation rates, fitness threshold, etc.) live in `config-feedforward.txt`.

## Files
- `main.py` — Entry point with the mode-selection menu
- `game_bird.py` — Bird/Pipe/Base classes and manual gameplay loop
- `game_neat.py` — NEAT training loop reusing the game entities
- `config-feedforward.txt` — NEAT configuration
- `imgs/` — Sprite assets: bird animation frames, pipes, background, base

## Running each mode standalone
Each file is also directly executable:
```bash
python game_bird.py     # manual only
python game_neat.py     # NEAT only
```
