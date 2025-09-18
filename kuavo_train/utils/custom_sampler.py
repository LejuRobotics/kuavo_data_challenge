import torch
from torch.utils.data import Sampler
from tqdm import tqdm

class EpisodeContextRateSampler(Sampler):
    def __init__(self, dataset, target_high_hz=30, target_low_hz=10, base_hz=30,
                 window_size=1.0, num_samples=None, high_threshold=0.7):
        """
        Args:
            dataset: dataset对象，每个样本是 dict，包括 'action' 和 'episode_index'
            target_high_hz: 高频帧目标采样率
            target_low_hz: 低频帧目标采样率
            base_hz: 数据原始采样率
            window_size: 高差分帧前后多少秒范围密集采样
            num_samples: 每个epoch采样总数
            high_threshold: 判定关键帧阈值
        """
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.high_rate = target_high_hz / base_hz
        self.low_rate = target_low_hz / base_hz
        self.window_frames = int(window_size * base_hz)

        # ---------- 1. 一次性提取 action 和 episode_index ----------
        actions_list = []
        episode_idx_list = []
        for i in tqdm(range(len(dataset)), desc="Extracting actions and episode indices"):
            sample = dataset[i]
            actions_list.append(sample['action'].unsqueeze(0))   # 保证 [1, action_dim]
            episode_idx_list.append(torch.tensor([sample['episode_index']], dtype=torch.long))

        actions = torch.cat(actions_list, dim=0)            # [N, action_dim]
        episode_index = torch.cat(episode_idx_list, dim=0)  # [N]
        print("Extracted actions and episode indices:", actions.shape, episode_index.shape)

        N = len(actions)
        diffs = torch.zeros(N, dtype=torch.float32)

        # ---------- 2. 计算差分，只计算同 episode 的相邻帧 ----------
        mask = episode_index[1:] == episode_index[:-1]
        d = (actions[1:] - actions[:-1]).pow(2).sum(dim=(1, 2))
        diffs[1:][mask] = d[mask]

        # ---------- 3. episode 内归一化 ----------
        diffs_norm = diffs.clone()
        for i, ep in tqdm(enumerate(episode_index.unique()), desc="Normalizing episode differences"):
            ep_mask = episode_index == ep
            ep_diffs = diffs[ep_mask]
            if len(ep_diffs) > 1:
                diffs_norm[ep_mask] = (ep_diffs - ep_diffs.min()) / (ep_diffs.max() - ep_diffs.min() + 1e-8)
            else:
                diffs_norm[ep_mask] = 0.0

        # ---------- 4. 初步权重映射 ----------
        weights = self.low_rate + (self.high_rate - self.low_rate) * diffs_norm

        # ---------- 5. 上下文扩展 ----------
        context_weights = weights.clone()
        high_indices = (diffs_norm > high_threshold).nonzero(as_tuple=True)[0]
        for hi in high_indices:
            start = max(0, hi - self.window_frames)
            end = min(N, hi + self.window_frames + 1)
            context_weights[start:end] = torch.maximum(
                context_weights[start:end],
                torch.full_like(context_weights[start:end], self.high_rate)
            )

        # ---------- 6. 归一化 ----------
        self.weights = context_weights / context_weights.sum()
        import matplotlib.pyplot as plt
        plt.plot(self.weights.numpy()[0:400])
        plt.title("Action Differences")
        plt.xlabel("Frame")
        plt.ylabel("Difference")
        plt.show()

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
