# kuavo_train

`kuavo_train` provides an encapsulation based on **LeRobot Diffusion Policy**, supporting depth imaging, Transformer architecture as well as fusing multimodal information. Through this encapsulation, the end user can freely construct and train their own customised policies with their own tasks.

## Characteristics

- **Diffusion Policy encapsulation**: Encapsulates training, configuration as well as inferencing, in an easy-to-use package.
- **Transformer support**: Policy natively supports Transformer architecture, enhancing the ability to model temporal features.
- **Depth Image Processing and Fusion:**: It can process depth images from multiple cameras and supports feature fusion.
- **Highly Expendable**: Other strategies can be customized by referencing existing encapsulations and through inheritance or modification.

## File Structure

```
kuavo_train/
├── utils/
│ ├── __init__.py
│ ├── augmenter.py # Data Augmenter
│ ├── transforms.py # Data augmentation based on lerobot, with transformconfigs
│ └── utils.py # Other assistive utilities, such as saving and loading random number states
├── wrapper/
│ ├── dataset/
│ │ └── LeRobotDatasetWrapper.py # Dataset encapsulation, provides only the dataset inheritence examples, which is not yet used
│ ├── policy/
│ │ └── diffusion/
│ │ ├── __init__.py
│ │ ├── DiffusionConfigWrapper.py   # diffusion config inheritence example
│ │ ├── DiffusionModelWrapper.py    # diffusion model inheritence example, with depth imaging and feature fusion support
│ │ ├── DiffusionPolicyWrapper.py   # diffusion policy inheritence example, containing input processing such as crop and resize
│ │ ├── DiT_1D_model.py             # DiT-based 1D data diffusion model, optional
│ │ ├── DiT_model.py                # DiT-based diffusion model, optional
│ │ └── transformer_diffusion.py    # Transformer-based diffusion model
│ └── __init__.py
├── README.md
├── train_policy.py                 # Start here for policy training
└── train_policy_with_accelerate.py # Start here for policy training with Accelerate (multi-GPU)
```

## How to Use

1. **Prepare Dataset**  
   Use `python kuavo_data/CvtRosbag2Lerobot.py` to convert dataset into Lerobot parquets.

2. **Config Policy**  
   Use `configs/policy/diffusion_config.yaml` to set up training, model and policy parameters.

3. **Train Policy**  
   Execute `python kuavo_train/train_policy.py` to begin training.

4. **Extending Towards Other Policies**  
   - Please refer to `DiffusionPolicyWrapper.py` for inheritence structures, and customise your own policy.
   - If such policy requires depth image processing as well as fusing multimodal information, refer to `DiffusionModelWrapper.py`.
   - Diffusion-based models can use existing Transformer modules here.

## Dependencies

- PyTorch
- Torchvision
- Other dependencies listed in `requirements_ilcode.txt`, as well as those listed in [README.md](../README.md)

---


---

### Model Training

Use the existing preconverted lerobot dataset to start imitation learning based training:

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

Details:

* `task`: Customise your own task name here (preferbly in line with the task name found in the data conversion, such as `pick and place`)
* `method`: Customise your own method name here to differentiate between different training attempts, such as `diffusion_bs128_usedepth_nofuse`
* `root`: Local path of your training data. Note that typically the `lerobot` folder name also needs to be present, and should be the same as the output path of step 1 above. I.e. `/path/to/lerobot_data/lerobot`
* `training.batch_size`: Batch size, adjust based on your GPU memory size
* `policy_name`: The policy type in use. Now supports only `diffusion` and `act` options
* Other parameters details can be found in the yaml files. It is recommended to directly modify such yaml files as to avoid command input typos

---

### Model Training with multi-GPU Setup

Install accelerate: pip install accelerate

```bash
accelerate launch --config_file ./configs/policy/accelerate_config.yaml \ 
  ./kuavo_train/train_policy_with_accelerate.py  --  \ 
  --config-path ./configs/policy \ 
  --config-name diffusion_config.yaml
```

Details:

* diffusion_config.yaml config options are the same as above.

---

> This directory is mainly geared towards experiments with robot IL-based policy, and helps you quickly construct diffusion-based policy templates.