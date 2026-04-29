# config修改简易教程
**下文讲的参数都是常改的**，一般修改这些就行，没讲的不知道怎么改不建议改，保持默认即可，知道怎么改可以按自己的需求修改

按数转——训练——推理的顺序

下面的标题是文件夹名
## data(改KuavoRosbag2Lerobot.yaml这个文件)
```
  rosbag_dir: /path/to/your/rosbag # rosbag文件存放的目录,建议使用绝对路径（在有bag的那一级文件夹打开终端，pwd出来的那个）

  lerobot_dir: /your/path/to/your/lerobotdata/  # 转换后的lerobot文件的保存目录，会在该目录下生成lerobot子目录，并存放对应数据，建议使用绝对路径

  platform_type: "4pro"  # 硬件平台类型，可选: "4pro", "5w" 或 "5"，默认使用 configs/platform/platform_config.yaml 中的 default 值

  eef_type: leju_claw # 末端执行器类型，仿真选择：rq2f85, 真机可选：leju_claw,（夹爪） qiangnao，（灵巧手）

  which_arm: right  # 需要哪一只手臂的关节 + 图像数据，可选: left, right, both，注意图像数据会同时包含头部相机图像

  use_depth: False  # 是否需要深度图像数据，与上面手臂对应的深度图像数据, 此分支act和dp均支持了深度图像，需要就转，不需要转了很浪费时间

  width: 848  # 图像缩放宽度，真机填848，仿真填640
```

## policy(用哪个policy改哪个，按照readme里的教程在启动训练的参数里选用的policy)

### act_config.yaml/diffusion_config.yaml
```
task: "your_task_name"  # 随便填，给你自己标识用的

method: "your_method_name"  # 随便填，给你自己标识用的

root: "/your/path/to/your/lerobotdata/lerobot"  # 转好的lerobot数据地址，建议绝对路径

max_epoch： 500  # 最大轮次，一般也不用训满，收敛了就差不多了

save_freq_epoch: 10  # 多少个epoch保存一次，不用设得太小，要不一直保存占空间

batch_size: 32  # 根据训练的终端性能来，设置成2的倍数

num_workers: 8  # # 根据训练的终端性能来，设置成2的倍数

custom:
    use_depth: true  # 训练是否使用深度，前面转了才能用
```

### gr00t_n1d5_config.yaml
```
task: "your_task_name"  # 随便填，给你自己标识用的

method: "your_method_name"  # 随便填，给你自己标识用的

root: "/your/path/to/your/lerobotdata/lerobot"  # 转好的lerobot数据地址，建议绝对路径

max_epoch： 500  # 最大轮次，一般也不用训满，收敛了就差不多了

save_freq_epoch: 10  # 多少个epoch保存一次，不用设得太小，要不一直保存占空间

batch_size: 32  # 根据训练的终端性能来，设置成2的倍数

num_workers: 8  # # 根据训练的终端性能来，设置成2的倍数

custom:
    action_head_mode: "single"  # 单臂填single，双臂写biped,填错了会崩
```

### pi0/pi05_config.yaml
```
task: "your_task_name"  # 随便填，给你自己标识用的

method: "your_method_name"  # 随便填，给你自己标识用的

root: "/your/path/to/your/lerobotdata/lerobot"  # 转好的lerobot数据地址，建议绝对路径

max_epoch： 500  # 最大轮次，一般也不用训满，收敛了就差不多了

save_freq_epoch: 10  # 多少个epoch保存一次，不用设得太小，要不一直保存占空间

batch_size: 32  # 根据训练的终端性能来，设置成2的倍数

num_workers: 8  # 根据训练的终端性能来，设置成2的倍数

use_wandb: false  # 是否用wandb看训练曲线，会用的写true自己注册帐号即可

custom:
    use_depth: true  # 训练是否使用深度，前面转了才能用
```
## accelerate（多卡训练的时候才需要）
```
num_processes: 2                # 用几张卡填几
gpu_ids: "6,7"                  # 指定使用第几号GPU
```

## deploy（改kuavo_env.yaml）
```
  env_name: Kuavo-Real  # 仿真Kuavo-Sim，真机Kuavo-Real

  eef_type: leju_claw  # 末端执行器类型: 仿真选择：rq2f85, 真机可选：leju_claw, qiangnao

  platform_type: "4pro"  # 硬件平台类型，可选: "4pro", "5w" 或 "5"

  which_arm: both  # 用哪个手，可选left， right, both

  image_size: &IMGSIZE [848, 480]  # 图像大小：宽，高, 真机848， 480；仿真640，480

  obs_key_map:  # 以下的这些，用哪些留哪些，不用的都注释了, 比如，用右手 + 夹爪不加深度，就把wrist_cam_l,三个深度，qiangnao，rq2f85注释了
    head_cam_h: ["/cam_h/color/image_raw/compressed", "CompressedImage", 30, *IMGSIZE]  # 这个必须要，别注释

    wrist_cam_l: ["/cam_l/color/image_raw/compressed", "CompressedImage", 30, *IMGSIZE]  

    wrist_cam_r: ["/cam_r/color/image_raw/compressed", "CompressedImage", 30, *IMGSIZE]  

    depth_h: ["/cam_h/depth/image_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE] 

    depth_l: ["/cam_l/depth/image_rect_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE]  
    
    depth_r: ["/cam_r/depth/image_rect_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE]  
    
    joint_q: ["/sensors_data_raw", "sensorsData", 500] # 这个得要，别注释

    qiangnao: ["/dexhand/state", "JointState", 500]  
    
    leju_claw: ["/leju_claw_state", "lejuClawState", 500] 
    rq2f85: ["/gripper/state", "JointState", 500]  

    inference:
        go_bag_path: /path/to/your/go.bag  # 真机推理时需要提供bag包的完整路径，如：到达预抓取姿态等，可直接拷贝一个训练的rosbag控制。

        policy_type: "act"  # 策略名字，支持diffusion，act,gr00t_n1d5等
        
        # 以下三个就是保存权重的三级文件夹的名字，直接去outputs/train里看就好
        task: "your_task"
        method: "your_method"
        timestamp: "your_timestamp" # 类似的运行时时间戳
        epoch: best  # 使用训练保存的哪一个epoch，可填50，100，best等，注意：代码将在outputs/train/<task>/<method>/<timestamp>/epoch<epoch>中load policy的模型参数

        max_episode_steps: 200  # 最大回合步数，超过自动结束，可根据任务所需时长调整

```