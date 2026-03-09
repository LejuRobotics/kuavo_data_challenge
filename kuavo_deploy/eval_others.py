import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!

from pathlib import Path

import gym_pusht  # noqa: F401
import gym_aloha
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.utils.random_utils import set_seed
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf,ListConfig
import random
from lerobot.policies.factory import make_pre_post_processors

@hydra.main(config_path="../configs/deploy/", config_name="others_env", version_base=None)
def main(cfg: DictConfig):

    set_seed(seed=cfg.seed)
    device = torch.device(cfg.device)
    # pretrained_policy_path = Path("outputs/train/")/ cfg.method / cfg.timestamp_epoch
    pretrained_policy_path = Path("outputs/train") / cfg.task / cfg.method / cfg.timestamp / f"epoch{cfg.epoch}"

    policy = CustomDiffusionPolicyWrapper.from_pretrained(pretrained_policy_path)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(None, Path(str(pretrained_policy_path).split("/epoch", 1)[0]))
    print("Processor",Path(str(pretrained_policy_path).split("/epoch", 1)[0]))

    # Create a directory to store the video of the evaluation
    output_directory = Path("outputs/eval/") / cfg.task / cfg.method / cfg.timestamp / f"epoch{cfg.epoch}"
    output_directory.mkdir(parents=True, exist_ok=True)
    
    env_obs_select = {"aloha":{"env": "gym_aloha/AlohaTransferCube-v0",  # gym_aloha/AlohaTransferCube-v0
                               "obs": "observation.images.top"},
                  "pusht": {"env": "gym_pusht/PushT-v0",
                               "obs": "observation.image"}
                    }[cfg.env]

    env = gym.make(
        env_obs_select["env"],
        obs_type="pixels_agent_pos",
        max_episode_steps=cfg.max_episode_steps,
    )


    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    print(policy.config.input_features)
    print(env.observation_space)

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    print(policy.config.output_features)
    print(env.action_space)

    # Log evaluation results
    log_file_path = output_directory / "evaluation_log.txt"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {cfg.eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(cfg.eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        numpy_observation, info = env.reset(seed=random.randint(0, 10000))

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []
        frames = []

        # Render frame of the initial state
        frames.append(env.render())

        step = 0
        done = False
        with tqdm(total=cfg.max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
            while not done:
                # Prepare observation for the policy
                state = torch.tensor(numpy_observation["agent_pos"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                # print(type(numpy_observation['pixels']))
                if isinstance(numpy_observation["pixels"],dict):
                    # print(f"Observations:{[k for k in numpy_observation['pixels'].keys()]}")
                    image = (torch.tensor(numpy_observation["pixels"]['top'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255).to(device, non_blocking=True)
                else:
                    image = (torch.tensor(numpy_observation["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255).to(device, non_blocking=True)


                # Create the policy input dictionary
                observation = {
                    "observation.state": state,
                    env_obs_select["obs"]: image,
                }
                observation = preprocessor(observation)
                # print(observation[env_obs_select["obs"]].max(), observation[env_obs_select["obs"]].min())
                # raise ValueError("Stop here")
                # Predict the next action with respect to the current observation
                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = postprocessor(action)
                # Prepare the action for the environment
                numpy_action = action.squeeze(0).to("cpu").numpy()
                if cfg.use_delta:
                    numpy_action = numpy_action + numpy_observation["agent_pos"]

                # Step through the environment and receive a new observation
                numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
                
                # Keep track of all the rewards and frames
                rewards.append(reward)
                frames.append(env.render())

                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                done = terminated | truncated | done
                step += 1
                
                # Update progress bar
                status = "Success" if terminated else "Running"
                pbar.set_postfix({
                    "Reward": f"{reward:.3f}",
                    "Status": status,
                    "Total Reward": f"{sum(rewards):.3f}"
                })
                pbar.update(1)

        if terminated:
            success_count += 1
            tqdm.write(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            tqdm.write(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.metadata["render_fps"]

        # Encode all frames into a mp4 video.
        video_path = output_directory / f"rollout_{episode+1}.mp4"
        imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / cfg.eval_episodes:.2f}\n")

    # Display final statistics
    print("\n" + "="*50)
    print(f"🎯 Evaluation completed!")
    print(f"📊 Success count: {success_count}/{cfg.eval_episodes}")
    print(f"📈 Success rate: {success_count / cfg.eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)



if __name__ =="__main__":
    main()