![LOOP driving simulation and learning dashboard](assets/simulation-screenshot-2026-11-25.png)

**Project by [Sabid (@sabidarrow)](https://github.com/sabidarrow)**

# LOOP — Reinforcement Learning Driving Lab

A runnable extension of the original NEAT car project. The new simulator uses **tabular Q-learning**, a procedural three-lane loop, mixed traffic, randomly arriving pedestrians and temporary barriers, and a native desktop dashboard. Python runs the simulation, learning, and graphics using Tkinter. No GPU, API key, internet connection, or third-party Python packages are needed with the standard Windows Python installer. An optional browser interface remains available.

## Quick start

Install Python 3.10 or newer, extract this project, open a terminal in its folder, and run:

```sh
python simulate.py
```

On Windows you can also double-click `START_WINDOWS.bat` (Python must be installed). On macOS/Linux use `python3` if needed. A desktop window opens with the road and learning dashboard together. Pause/resume, change simulation speed, and save the model using the controls. Close the window to save the model and stop. Space toggles pause; Escape closes the window. Speed presets are approximate and fast training is bounded to keep controls responsive. Tkinter is included with the standard Windows Python installer; some Linux installations require the distribution's `python3-tk` package.

### Run from Spyder — no browser

1. Stop the previous browser simulation with Spyder's Stop button (or Ctrl+C in the terminal that started it).
2. Open `run_in_spyder.py` in Spyder and press F5.
3. The road, charts, statistics, and controls appear in one native Python window. Close it to save and return to Spyder's console.

The window is separate from Spyder's editor and Plots pane, but is run entirely by Python in Spyder's session. No web server starts. The launcher resolves imports from its own folder and ignores unrelated IPython command-line arguments. Use a Spyder interpreter with Tkinter available. You can also run `simulate.py` directly; the default is now desktop mode.

Optional isolated Python environment on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe simulate.py
```

On macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python simulate.py
```

There are no packages to install for the new simulator. `requirements.txt` documents this. The original `arrow.py`, `config.txt`, image maps, and notebook are preserved; the original program still needs its own `pygame` and `neat-python` dependencies.

## What happens in the world

- A 600-metre, three-lane closed circuit, with roadside trees drawn procedurally.
- One green **learning autonomous vehicle** chooses hold, acceleration, braking, left lane change, or right lane change. It starts a fresh episode after a collision or 120 simulated seconds, retaining its Q-table.
- Six background vehicles include grey human traffic, purple scripted autonomous cars, and blue connected cars. Background drivers follow a simple gap-based speed controller; they do not learn.
- Pedestrians cross the road at randomly sampled locations and walking speeds. Temporary stationary barriers appear and expire. Spawn rejection prevents objects appearing within 35 metres of a vehicle. These are stochastic synthetic events, not calibrated real-world traffic distributions.
- Local sensors see objects within 45 m. With V2X enabled, a blue vehicle within 150 m relays objects within its own 90 m sensing radius. Dashed blue links show relay connectivity. Messages expose object positions, not an automatic collision-avoidance override.
- Live episode return, speed, distance, exploration, visited states, temporal-difference error, recent crash rate, event log, and return chart. The chart shows this session's most recent episodes. CSV logs persist between launches.

## Train, save, resume, and evaluate

To watch the supplied 1,000-episode checkpoint immediately, run `python simulate.py --watch --model examples/demo_policy.json`. See `BENCHMARK.md` for measured results and limitations. The packaged default launch starts a fresh policy; the demo loads only when requested.

```sh
# Train 3,000 additional episodes, without rendering
python simulate.py --train --episodes 3000

# Watch a frozen policy without exploration or Q-table updates
python simulate.py --watch

# Compare random, rule-based, and learned policies on 100 held-out seeds
python simulate.py --evaluate --episodes 100

# Train a separate policy without connected-vehicle messages
python simulate.py --train --episodes 3000 --no-v2x --model runs/no_v2x.json --output runs/no_v2x

# Watch that policy without V2X
python simulate.py --watch --no-v2x --model runs/no_v2x.json

# Reproducible fresh experiment (choose a new model path)
python simulate.py --train --episodes 1000 --seed 123 --model runs/seed123.json --output runs/seed123

# Optional browser interface (desktop is the default)
python simulate.py --browser

# Serve the optional browser interface without automatically opening a tab
python simulate.py --browser --no-browser --port 8766
```

The default checkpoint is `runs/q_policy.json`. Existing checkpoints load automatically; `--episodes` is the number of **additional** training episodes. Save every 25 episodes, on the Save button, and on Ctrl+C. A resumed run retains Q-values and the exploration schedule, but starts a fresh world/random sequence rather than continuing mid-episode. For a fresh policy use a new `--model` path.

`runs/episodes.csv` records episode return, distance, collision type, near misses, epsilon, mode, and seed. `runs/evaluation.csv` contains per-episode held-out results; `runs/evaluation.json` summarizes reward, distance, collision rate and crashes/km. Evaluation freezes learning and uses seeds offset by 100,000. Each policy starts with the same scenario seeds; later accepted spawn events can differ because spawn safety depends on vehicle positions. Evaluation overwrites its prior result files, so use separate output directories for separate experiments.

The built-in evaluation tests the **same checkpoint** with V2X on and off. For a stronger connected-mobility experiment, train separate V2X and non-V2X policies with equal training budgets, evaluate both, and compare their matching V2X configurations across several training seeds. Report uncertainty across seeds and compare with the rule-based baseline, not only random driving.

## Learning method

State: current lane, three speed bins, four front-clearance bins for each lane, rear-occupancy flags for each lane, and target lane. Predicted pedestrian crossing occupancy covers the next two seconds. Actions: hold, accelerate at 3 m/s², brake at 6 m/s², and change one lane left/right. Lane changes take one second; speed is capped at 17 m/s. Dynamics run at 0.2-second decisions with four collision-checking substeps. The road centreline is followed automatically; this is tactical lane/speed learning rather than end-to-end steering or vision.

```
Q(s,a) <- Q(s,a) + 0.12 * [r + 0.97 * max Q(s',a') - Q(s,a)]
```

Terminal transitions use no future-value term. Epsilon decays exponentially from 0.95 toward 0.04 over episodes. Reward is distance / 3.4 minus a 0.15 time cost, 0.15 lane-change cost, 0.5 invalid-action cost, 0.5 proximity cost, and a 100-point collision penalty. Progress rewards discourage permanent stopping; collision penalties discourage reckless progress. Learning updates every action, including crashes. No rule forces the learned action to be safe.

The discretized observation is partially observable and uses a compact approximation. Tabular Q-learning does not guarantee steadily improving scores or collision-free driving here. Watch mode disables exploration; training-mode crash rates include deliberate random exploration. Evaluate distance alongside crash rate so a policy that merely stops cannot look successful.

## Project value and limitations

This is a useful educational and portfolio project because it combines a reproducible environment, actual experience-based learning, connected-vehicle sensing, interpretable telemetry, and baseline comparisons. A defensible research question is: **Does shared hazard information improve distance travelled and reduce collisions under the same training budget?**

It is not a validated road-safety simulator or a deployable autonomous-driving system. Trees are decorative; background traffic is scripted; only learner collisions are scored; the model has no intersections, realistic vehicle dynamics, occlusion, communication delay/loss, or calibrated pedestrian behaviour. One learner is intentional; it is not multi-agent RL. Extensions for research include multiple learning cars, packet loss and latency, more realistic traffic, richer observations, and DQN/PPO comparisons. Do not infer real-world safety from this simulation.

## Verify and project layout

```sh
python -m unittest discover -s tests -v
```

- `run_in_spyder.py`: F5 launcher for the native desktop window.
- `simulate.py`: CLI, experiments, logging, optional localhost dashboard server.
- `loop_sim/desktop.py`: native Tkinter animation, charts, controls, and telemetry.
- `loop_sim/core.py`: world dynamics, V2X observations, rewards, Q-learning.
- `loop_sim/dashboard.html`: local visualization and charts.
- `tests/test_simulation.py`: deterministic dynamics, collision, V2X, checkpoint and learning checks.
- `arrow.py`, `config.txt`, original maps/notebook: preserved original project.

Desktop mode starts no server. The optional browser server binds only to localhost. There are no external assets or data uploads. Desktop tests instantiate withdrawn Tk windows and require a graphical desktop; on headless machines run `python -m unittest discover -s tests -p test_simulation.py` for dynamics-only tests.
