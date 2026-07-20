"""生成 Wheeled-SRL 训练指标图（四象限，对应论文 tab:training_metrics）"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth(y, w=50):
    """滑动平均"""
    if len(y) < w: return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='same')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(args.log_dir, 'training_metrics.png')

    ea = EventAccumulator(args.log_dir)
    ea.Reload()

    def get(tag):
        events = ea.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ── (1) 平均奖励 ──
    x, y = get('Train/mean_reward')
    axs[0, 0].plot(x, y, 'b-', alpha=0.3, lw=0.5, label='Raw')
    axs[0, 0].plot(x, smooth(y), 'b-', lw=1.5, label='Smoothed')
    axs[0, 0].set_ylabel('Mean Reward')
    axs[0, 0].set_title('(a) Average Reward')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, alpha=0.3)

    # ── (2) 回合长度 ──
    x, y = get('Train/mean_episode_length')
    axs[0, 1].plot(x, y, 'g-', alpha=0.3, lw=0.5, label='Raw')
    axs[0, 1].plot(x, smooth(y), 'g-', lw=1.5, label='Smoothed')
    axs[0, 1].set_ylabel('Episode Length (steps)')
    axs[0, 1].set_title('(b) Episode Length')
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True, alpha=0.3)

    # ── (3) 奖励分量分解 ──
    reward_tags = {
        'jump_height': ('Jump Height', 'C0'),
        'landing_soft': ('Landing Soft', 'C1'),
        'crouch_prep': ('Crouch Prep', 'C2'),
        'wheel_ground_matching': ('Wheel-Ground Match', 'C3'),
        'tracking_lin_vel': ('Lin Vel Track', 'C4'),
        'tracking_ang_vel': ('Ang Vel Track', 'C5'),
        'vertical_thrust': ('Vertical Thrust', 'C6'),
        'crouch_depth': ('Crouch Depth', 'C7'),
    }
    for tag, (label, color) in reward_tags.items():
        try:
            x, y = get(f'reward/{tag}')
            axs[1, 0].plot(x, smooth(y), color=color, lw=0.8, label=label, alpha=0.8)
        except: pass
    axs[1, 0].set_ylabel('Reward Value')
    axs[1, 0].set_title('(c) Reward Components')
    axs[1, 0].legend(fontsize=7, ncol=2)
    axs[1, 0].grid(True, alpha=0.3)

    # ── (4) Action Std + FPS ──
    x1, y1 = get('Policy/mean_std')
    ax2 = axs[1, 1]
    ax2.plot(x1, y1, 'r-', lw=1.0, label='Action Std')
    ax2.set_ylabel('Action Std', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax2b = ax2.twinx()
    try:
        x2, y2 = get('Perf/total_fps')
        ax2b.plot(x2, y2, 'purple', lw=1.0, alpha=0.7, label='FPS')
    except: pass
    ax2b.set_ylabel('FPS', color='purple')
    ax2b.tick_params(axis='y', labelcolor='purple')
    ax2.set_title('(d) Action Std & Training Speed')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    for ax in axs.flat:
        ax.set_xlabel('Iterations')

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.out}")

if __name__ == '__main__':
    main()
