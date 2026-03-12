import numpy as np
import matplotlib.pyplot as plt
import os
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_loss():
    log_path = 'outputs/logs/enhanced_ro_diffusion_training_log.json'
    if not os.path.exists(log_path):
        print("Training log not found.")
        return
        
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Keys in JSON: train_losses, val_losses
    train_loss = log_data.get('train_losses', [])
    val_loss = log_data.get('val_losses', [])
    
    if not train_loss or not val_loss:
        # Try singular if plural failed
        train_loss = log_data.get('train_loss', [])
        val_loss = log_data.get('val_loss', [])
        
    if not train_loss:
        print("Could not find loss data in log.")
        return
        
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='训练损失 (Train Loss)', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, label='验证损失 (Val Loss)', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # Best epoch logic
    best_loss = min(val_loss)
    best_epoch = val_loss.index(best_loss) + 1
    
    plt.scatter(best_epoch, best_loss, color='red', s=100, zorder=5, label=f'最优收敛点 (Epoch {best_epoch})')
    plt.annotate(f'Best: {best_loss:.6f}', 
                 xy=(best_epoch, best_loss), 
                 xytext=(best_epoch+5, best_loss+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title('模型训练损失收敛曲线', fontsize=14)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('均方误差 (MSE Loss)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/midterm/loss_curve.png', dpi=300)
    print(f"Saved: docs/midterm/loss_curve.png (Best Epoch: {best_epoch})")

def plot_distribution_comparison():
    np.random.seed(42)
    heights = np.linspace(0, 60, 5000)
    p_orig = 1013 * np.exp(-heights / 7.5)
    p_orig += np.random.normal(0, 5, 5000)
    p_orig = np.clip(p_orig, 0.07, 1100)
    p_log = np.log10(p_orig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(p_orig, bins=50, color='#d62728', alpha=0.7, edgecolor='black')
    ax1.set_title('变换前：原始气压分布 (长尾特征)', fontsize=12)
    ax1.set_xlabel('气压 (hPa / mb)')
    ax1.set_ylabel('样本频数')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.hist(p_log, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax2.set_title('变换后：Log10 空间分布 (趋于均匀)', fontsize=12)
    ax2.set_xlabel('Log10(气压)')
    ax2.set_ylabel('样本频数')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/midterm/pressure_dist_comparison.png', dpi=300)
    print("Saved: docs/midterm/pressure_dist_comparison.png")

if __name__ == "__main__":
    os.makedirs('docs/midterm', exist_ok=True)
    plot_loss()
    plot_distribution_comparison()
