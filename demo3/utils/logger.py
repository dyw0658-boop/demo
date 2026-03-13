"""
日志记录工具模块
"""

import os
import time
import json
import torch
import pandas as pd

# 尝试导入tensorboard，如果失败则提供替代方案
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard不可用，将使用简化日志记录")
    
    # 创建一个简化的SummaryWriter替代类
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_image(self, *args, **kwargs):
            pass
        def add_images(self, *args, **kwargs):
            pass
        def add_histogram(self, *args, **kwargs):
            pass
        def add_text(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass


class Logger:
    """
    多功能日志记录器
    """
    
    def __init__(self, log_dir='./logs', tensorboard=True, csv_log=True):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            tensorboard: 是否使用 TensorBoard
            csv_log: 是否保存 CSV 日志
        """
        self.log_dir = log_dir
        self.tensorboard = tensorboard
        self.csv_log = csv_log
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard 写入器
        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        else:
            self.writer = None
        
        # CSV 日志
        if csv_log:
            self.csv_file = os.path.join(log_dir, 'training_log.csv')
            self.csv_data = []
        
        # 训练开始时间
        self.start_time = time.time()
        
        # 日志缓存
        self.log_cache = {}
    
    def log_scalar(self, tag, value, step):
        """
        记录标量值
        
        Args:
            tag: 标签
            value: 值
            step: 步数
        """
        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        
        # CSV 日志
        if self.csv_log:
            if step not in self.log_cache:
                self.log_cache[step] = {'step': step}
            self.log_cache[step][tag] = value
    
    def log_scalars(self, tag_value_dict, step):
        """
        记录多个标量值
        
        Args:
            tag_value_dict: 标签-值字典
            step: 步数
        """
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)
    
    def log_images(self, tag, images, step, nrow=8):
        """
        记录图像
        
        Args:
            tag: 标签
            images: 图像张量
            step: 步数
            nrow: 每行图像数
        """
        if self.writer is not None:
            # 创建图像网格
            grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
            self.writer.add_image(tag, grid, step)
    
    def log_histogram(self, tag, values, step):
        """
        记录直方图
        
        Args:
            tag: 标签
            values: 值
            step: 步数
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_parameters(self, model, step):
        """
        记录模型参数
        
        Args:
            model: 模型
            step: 步数
        """
        if self.writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"parameters/{name}", param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_config(self, config):
        """
        记录配置
        
        Args:
            config: 配置字典
        """
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.writer is not None:
            self.writer.add_text('config', json.dumps(config, indent=2))
    
    def save_checkpoint(self, state, filename):
        """
        保存检查点
        
        Args:
            state: 状态字典
            filename: 文件名
        """
        checkpoint_file = os.path.join(self.log_dir, filename)
        torch.save(state, checkpoint_file)
    
    def flush(self):
        """刷新日志"""
        # 保存 CSV 日志
        if self.csv_log and self.log_cache:
            # 转换为 DataFrame
            df = pd.DataFrame(list(self.log_cache.values()))
            
            # 如果文件存在，追加数据
            if os.path.exists(self.csv_file):
                existing_df = pd.read_csv(self.csv_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # 保存到 CSV
            df.to_csv(self.csv_file, index=False)
            
            # 清空缓存
            self.log_cache = {}
    
    def close(self):
        """关闭日志记录器"""
        # 刷新日志
        self.flush()
        
        # 关闭 TensorBoard 写入器
        if self.writer is not None:
            self.writer.close()
        
        # 计算总训练时间
        total_time = time.time() - self.start_time
        print(f"训练完成，总时间: {total_time:.2f} 秒")
    
    def get_elapsed_time(self):
        """获取已用时间"""
        return time.time() - self.start_time


class ProgressLogger:
    """
    进度日志记录器
    """
    
    def __init__(self, total_steps, description="Training"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step, metrics=None):
        """
        更新进度
        
        Args:
            step: 当前步数
            metrics: 指标字典
        """
        self.current_step = step
        
        # 计算进度
        progress = step / self.total_steps
        elapsed_time = time.time() - self.start_time
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        
        # 格式化输出
        progress_bar = f"[{'>' * int(progress * 20):<20}] {progress:.1%}"
        time_info = f"Elapsed: {elapsed_time:.0f}s, ETA: {eta:.0f}s"
        
        # 添加指标信息
        metrics_info = ""
        if metrics:
            metrics_info = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # 打印进度
        print(f"\r{self.description}: {progress_bar} | {time_info} | {metrics_info}", end="")
        
        # 如果完成，换行
        if step >= self.total_steps:
            print()
    
    def close(self):
        """关闭进度记录器"""
        total_time = time.time() - self.start_time
        print(f"{self.description} 完成，总时间: {total_time:.2f} 秒")


def setup_logging(config, experiment_name=None):
    """
    设置日志记录
    
    Args:
        config: 配置字典
        experiment_name: 实验名称
    
    Returns:
        logger: 日志记录器实例
    """
    # 生成实验名称
    if experiment_name is None:
        experiment_name = f"experiment_{int(time.time())}"
    
    # 创建日志目录
    log_dir = os.path.join(config['logging']['save_dir'], experiment_name)
    
    # 创建日志记录器
    logger = Logger(
        log_dir=log_dir,
        tensorboard=config['logging'].get('tensorboard', True),
        csv_log=True
    )
    
    # 记录配置
    logger.log_config(config)
    
    return logger


if __name__ == "__main__":
    # 测试代码
    config = {
        'model': {'name': 'test_model'},
        'training': {'epochs': 10}
    }
    
    logger = setup_logging(config, 'test_experiment')
    
    # 记录一些数据
    for step in range(100):
        logger.log_scalar('loss', 1.0 / (step + 1), step)
        logger.log_scalar('accuracy', step / 100.0, step)
    
    logger.close()
    print("日志记录器测试完成")