# Kuavo Robot Control Examples

> A ROS-based Kuavo robot control example project. It supports arm motion control, trajectory replay, and policy/model inference.

## File Structure

```
kuavo_deploy/src/
├── eval/               # Evaluation scripts
│   ├── eval_kuavo.py   # Kuavo environment evaluation script
│   └── auto_test/      # Automated tests
│       ├── eval_kuavo.py           # Automated evaluation for Kuavo env
│       └── eval_kuavo_autotest.py  # Auto-test entry
└── scripts/            # Control scripts
    ├── script.py       # Main control script
    ├── controller.py   # Remote controller (send commands)
    └── script_auto_test.py  # Automated control script
```

## System Overview

The Kuavo control system includes these core components:

1. **`script.py`** - Main control script that runs robot tasks
2. **`controller.py`** - Remote controller to send commands to a running task
3. **`eval_kuavo.py`** - Evaluation script for inference and performance evaluation
4. **`script_auto_test.py`** - Batch/automated control script for repeated testing

## Quick Start

### Option 1: Use the interactive script `eval_kuavo.sh`

Start the interactive control interface:

```bash
bash kuavo_deploy/eval_kuavo.sh
```

You will see prompts like:

```bash
=== Kuavo Robot Control Examples ===
This script shows how to run different tasks via CLI arguments
-e supports pause/resume/stop

Control features:
  Pause/Resume: send SIGUSR1 (kill -USR1 <PID>)
  Stop task:    send SIGUSR2 (kill -USR2 <PID>)
  View logs:    tail -f log/kuavo_deploy/kuavo_deploy.log

kuavo_deploy/eval_kuavo.sh: 16: Bad substitution
1. Show help:
python kuavo_deploy/src/scripts/script.py --help

2. Dry run - preview what will be executed:
python kuavo_deploy/src/scripts/script.py --task go --dry_run --config /path/to/custom_config.yaml

3. Move to the working position:
python kuavo_deploy/src/scripts/script.py --task go --config /path/to/custom_config.yaml

4. Run the model from the current state:
python kuavo_deploy/src/scripts/script.py --task run --config /path/to/custom_config.yaml

5. Interpolate to the last frame of the bag, then start running:
python kuavo_deploy/src/scripts/script.py --task go_run --config /path/to/custom_config.yaml

6. Start from the last frame of the go_bag:
python kuavo_deploy/src/scripts/script.py --task here_run --config /path/to/custom_config.yaml

7. Return to zero pose:
python kuavo_deploy/src/scripts/script.py --task back_to_zero --config /path/to/custom_config.yaml

8. Auto-test in simulation, for eval_episodes runs:
python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config /path/to/custom_config.yaml

9. Enable verbose output:
python kuavo_deploy/src/scripts/script.py --task go --verbose --config /path/to/custom_config.yaml

=== Task Descriptions ===
go           - Interpolate to the first frame of the bag, then replay the bag to reach the working position
run          - Run the model from the current position
go_run       - Move to the working position, then run the model
here_run     - Interpolate to the last frame of the bag, then start running
back_to_zero - After interrupting inference, play the bag backwards to return to zero pose
auto_test    - Auto-test in simulation, run eval_episodes times

Choose: 1. Show normal help 2. Show auto-test help 3. Choose example
1. Run: python kuavo_deploy/src/scripts/script.py --help
2. Run: python kuavo_deploy/src/scripts/script_auto_test.py --help
3. Choose example
Select (1-3) or press Enter to exit:
```

Enter `3` and press Enter, then you will be prompted:

```bash
Please input the path to your custom config file:
```

Provide the config path (see the default config `configs/deploy/kuavo_sim_env.yaml`), then you will see:

```bash
Config path: configs/deploy/kuavo_sim_env.yaml
Parsing config...
Model config:
   Task: your_task
   Method: your_methof
   Timestamp: your_timestamp
   Epoch: 300
Full model path: your_path
Model path exists
Available examples:
1. Interpolate to the first frame, replay the bag to working pose (dry run)
Run: python kuavo_deploy/src/scripts/script.py --task go --dry_run --config /path/to/config.yaml
2. Interpolate to the first frame, replay the bag to working pose
Run: python kuavo_deploy/src/scripts/script.py --task go --config /path/to/config.yaml
3. Run the model from the current state
Run: python kuavo_deploy/src/scripts/script.py --task run --config /path/to/config.yaml
4. Move to the working position and run the model
Run: python kuavo_deploy/src/scripts/script.py --task go_run --config /path/to/config.yaml
5. Interpolate to the last frame of the bag and start running
Run: python kuavo_deploy/src/scripts/script.py --task here_run --config /path/to/config.yaml
6. Return to zero pose
Run: python kuavo_deploy/src/scripts/script.py --task back_to_zero --config /path/to/config.yaml
7. Same as (2) with verbose output
Run: python kuavo_deploy/src/scripts/script.py --task go --verbose --config /path/to/config.yaml
8. Auto-test in simulation, for eval_episodes runs
Run: python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config /path/to/config.yaml
9. Exit
Select (1-9)
```

Choose the feature you need. Usually, pick `8` to run automated tests in simulation.

The interactive script provides:
- List of available command examples
- Interactive task selection
- Live task control (pause/resume/stop)
- Live log viewing

Note: To run auto-test in the simulation environment, start `roscore` on your machine first, then start the auto-test script in `kuavo-ros-opensource`, and finally start this script.

#### Supported Task Types

| Task | Description | Typical Use |
|------|-------------|-------------|
| `go` | Interpolate to the first frame of the bag and replay it to the working position | Preparation |
| `run` | Run the model from the current state | Quick test |
| `go_run` | Move to the working position and run the model | Full pipeline |
| `here_run` | Interpolate to the last frame of the bag and start running | Continuous inference |
| `back_to_zero` | After interrupting inference, replay the bag backwards to zero pose | Safe rollback |
| `auto_test` | Run repeated tests in simulation and evaluate performance | Batch testing |

### Option 2: Run the Python scripts directly

#### 1) Show help

```bash
python kuavo_deploy/src/scripts/script.py --help
```

#### 2) Run basic tasks

```bash
# Interpolate to the first frame of the bag and replay it to the working position
python kuavo_deploy/src/scripts/script.py --task go --config /path/to/config.yaml

# Run the model from the current state
python kuavo_deploy/src/scripts/script.py --task run --config /path/to/config.yaml

# Move to the working position and run the model
python kuavo_deploy/src/scripts/script.py --task go_run --config /path/to/config.yaml

# Interpolate to the last frame of the bag and start running
python kuavo_deploy/src/scripts/script.py --task here_run --config /path/to/config.yaml

# Return to zero pose
python kuavo_deploy/src/scripts/script.py --task back_to_zero --config /path/to/config.yaml

# Run auto-test (simulation)
python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config /path/to/config.yaml
```

#### 3) Remote control while a task is running (`controller.py`)

`controller.py` provides a friendlier remote control interface:

```bash
# Basic usage
python kuavo_deploy/src/scripts/controller.py <command>

# Available commands
python kuavo_deploy/src/scripts/controller.py pause    # pause
python kuavo_deploy/src/scripts/controller.py resume   # resume
python kuavo_deploy/src/scripts/controller.py stop     # stop
python kuavo_deploy/src/scripts/controller.py status   # status

# Control a specific process
python kuavo_deploy/src/scripts/controller.py pause --pid 12345
```

Features of `controller.py`:

- Auto process discovery: finds running `script.py` processes
- Precise control: supports controlling a specific PID
- Status monitoring: shows process details (CPU, memory, runtime, etc.)
- Safety checks: verifies the target is a valid `script.py` process

#### 4) CLI arguments

##### `script.py`

Required:
- `--task`: task type (`go`, `run`, `go_run`, `here_run`, `back_to_zero`)
- `--config`: path to the config file

Optional:
- `--verbose`, `-v`: verbose output
- `--dry_run`: dry run mode (print actions only)

##### `script_auto_test.py`

Required:
- `--task`: task type (`auto_test`)
- `--config`: path to the config file

Optional:
- `--verbose`, `-v`: verbose output
- `--dry_run`: dry run mode (print actions only)

##### `controller.py`

Required:
- `command`: control command (`pause`, `resume`, `stop`, `status`)

Optional:
- `--pid`: target PID (if omitted, the script will auto-discover)

## Configuration

Default config file: `configs/deploy/kuavo_sim_env.yaml`

### Key Fields

```yaml
# 1) Environment config (aligned with configs/deploy/kuavo_sim_env.yaml)
real: false                   # use real robot or not
only_arm: true                # use arm-only data
eef_type: rq2f85              # end-effector type: qiangnao, leju_claw, rq2f85
control_mode: joint           # control mode: joint / eef
which_arm: both               # arm selection: left, right, both
head_init: [0, 0.209]         # initial head angles
input_images: ["head_cam_h", "wrist_cam_r", "wrist_cam_l", "depth_h", "depth_r", "depth_l"]
image_size: [480, 640]        # image size
ros_rate: 10                  # inference rate (Hz)

# Advanced config (not recommended to modify)
qiangnao_dof_needed: 1        # qiangnao DOF needed: 1 = open/close only
leju_claw_dof_needed: 1       # claw DOF needed
rq2f85_dof_needed: 1          # rq2f85 DOF needed
arm_init: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
arm_min: [-180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180]
arm_max: [ 180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180,  180]
eef_min: [0]
eef_max: [1]
is_binary: false

# 2) Inference config
go_bag_path: /path/to/your/bag/file.bag  # rosbag path

policy_type: "diffusion"
use_delta: false
eval_episodes: 1
seed: 42
start_seed: 42
device: "cuda"  # or "cpu"

# Model path: outputs/train/{task}/{method}/{timestamp}/epoch{epoch}
task: "ruichen"                  # replace with your training task
method: "test_git_model"         # replace with your training method
timestamp: "run_20250819_115313" # replace with your timestamp
epoch: 29                         # replace with your epoch

max_episode_steps: 500
env_name: Kuavo-Real
```

### End-Effector Configuration

| Type | Notes | DOF | Control |
|------|-------|-----|---------|
| `qiangnao` | Dexterous hand | 1 | open/close only |
| `leju_claw` | Gripper | 1 | close/open |

## Requirements

- ROS environment is configured
- Robot hardware is connected and healthy
- Config file path is correct
- Model files are complete
- Python dependencies are installed

## Troubleshooting

### Common Issues

| Issue | Fix |
|------|-----|
| Config file not found | Verify the config path |
| Arm initialization failed | Check ROS setup and hardware connection |
| Model path not found | Verify the model path in the config |
| `controller.py` cannot find the process | Ensure `script.py` is running, or specify `--pid` |
| Permission denied | Use `sudo` or check process permissions |
| High failure rate in auto-test | Check model quality; adjust `eval_episodes` |

### Debug Tips

1. Prefer `--dry_run` on first use
2. Verify robot hardware state
3. Use `python kuavo_deploy/src/scripts/controller.py status` to inspect task state
4. Check `log/kuavo_deploy/kuavo_deploy.log` for detailed logs

## Logging

- `log_model`: network/model logs
- `log_robot`: robot control logs

Log file location: `log/kuavo_deploy/kuavo_deploy.log`

## Safety Notes

1. Prefer `--dry_run` on first use
2. Verify robot hardware state
3. Emergency stop: `Ctrl+C` and `kill -USR2`
4. Validate paths and parameters in the config
5. Ensure you have sufficient permissions to control the target process
6. Monitor task status regularly

## Extending

To add a new task type:

1. Add a new method in the `ArmMove` class
2. Add a new option in `parse_args()`
3. Add the mapping in `task_map`
4. Update docs and examples

To extend control commands:

1. Add a new command in `controller.py`
2. Add the corresponding signal handling in `script.py`
3. Update the help text and examples

## Best Practices

### Recommended Workflow

1. Validate config → use `--dry_run`
2. Start a task → run `run_example.sh`
3. Diagnose issues → check logs
4. Exit safely → run `back_to_zero`

### Performance Tips

- Use `--verbose` for debugging; disable it in production
- Tune `ros_rate` to balance stability and performance
- Clean up logs periodically to avoid disk pressure
- Prefer `kuavo_deploy/src/scripts/controller.py` instead of manually killing processes
- For auto-test, choose a reasonable `eval_episodes` and validate in simulation first

---
