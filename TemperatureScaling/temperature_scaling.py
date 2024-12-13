import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


def nll_loss_with_temperature_all_class(T):
    """计算负对数似然损失 (NLL) ，所有的类别都使用同一个温度参数"""

    T = float(T)  # 温度参数
    scaled_logits = logits_tensor / T  # 使用给定的 T 对 logits 进行 scaling
    log_probs = F.log_softmax(scaled_logits, dim=1)  # 重新计算概率
    loss = F.nll_loss(log_probs, labels_tensor)  # 计算负对数似然损失
    return loss.item()


def nll_loss_with_temperature_per_class(T, class_idx):
    """计算每个类别的负对数似然损失 (NLL) ，每个类别使用不同的温度参数"""

    T = float(T)  # 温度参数
    scaled_logits = logits_tensor / T  # 使用给定的 T 对 logits 进行缩放
    log_probs = F.log_softmax(scaled_logits, dim=1)  # 计算经过 softmax 的概率分布
    loss = F.nll_loss(log_probs[:, class_idx], (labels_tensor == class_idx).long())  # 只计算当前类别的负对数似然损失
    return loss.item()


def apply_temperature_scaling_all_class(logits, T_optimal):
    """使用所有类别的最优温度参数 T_optimal 来标定 logits"""

    scaled_logits = logits / T_optimal  # 使用所有类别的最优温度参数 T_optimal 来标定 logits
    calibrated_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)  # 计算标定后的概率
    return calibrated_probs


def apply_temperature_scaling_per_class(logits, T_optimal_per_class):
    """使用每个类别的最优温度参数 T_optimal_per_class 来标定 logits"""

    scaled_logits = np.zeros_like(logits)  # 初始化 scaled_logits
    for class_idx, T in enumerate(T_optimal_per_class):
        scaled_logits[:, class_idx] = logits[:, class_idx] / T  # 使用每个类别的最优温度参数 T_optimal_per_class 来标定 logits
    calibrated_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)  # 计算标定后的概率
    return calibrated_probs


if __name__ == '__main__':
    # 假设我们有一个经过训练的模型，并且已经通过验证集得到了模型输出的 logits 和实际标签
    # logits 是模型在验证集上未经过 softmax 的输出 (shape: [num_samples, num_classes])
    # labels 是验证集的实际标签 (shape: [num_samples])

    logits = np.load('logits.npy')  # 导入 logits
    labels = np.load('labels.npy')  # 导入 labels

    class_list = ['Label_A', 'Label_B', 'Label_C']  # 定义类别列表
    num_classes = len(class_list)  # 获取类别数量
    class_dict = dict(zip(class_list, range(len(class_list))))  # 创建类别字典

    # 将 logits 和 labels 转换为 PyTorch 张量
    logits_tensor = torch.tensor(logits).type(torch.float32).to('cuda')  # 将 logits 转换为 PyTorch 张量
    labels_tensor = torch.tensor(np.array([class_dict[label] for label in labels]), dtype=torch.long).to('cuda')  # 将 labels 转换为数值，变为 PyTorch 张量

    adjust_mode = 'per_class'  # 选择调整模式

    if adjust_mode == 'all_class':
        T_optimal = minimize(nll_loss_with_temperature_all_class, x0=0.5, bounds=[(0.05, 2)], method='Nelder-Mead',
                             options={'maxiter': 1000})  # 使用 Nelder-Mead 方法优化温度参数
        T_optimal = T_optimal.x[0]  # 获取最优温度参数

        calibrated_probs = apply_temperature_scaling_all_class(logits, T_optimal)  # 应用温度标定到 logits

    elif adjust_mode == 'per_class':
        T_optimal_per_class = []

        for class_idx in range(num_classes):
            result = minimize(nll_loss_with_temperature_per_class, x0=0.5, bounds=[(0.05, 2)], method='Nelder-Mead',
                              options={'maxiter': 1000})  # 使用 Nelder-Mead 方法优化温度参数
            T_optimal_per_class.append(result.x[0])  # 将每个类别的最优温度参数添加到列表中

        calibrated_probs = apply_temperature_scaling_per_class(logits, T_optimal_per_class)  # 应用温度标定到 logits

    else:
        raise ValueError(f"Invalid adjust_mode: {adjust_mode}")

    # 对比原始概率和标定后的概率
    original_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # 原始概率
    original_max_probs = np.max(original_probs, axis=1)  # 原始概率的最大值
    calibrated_max_probs = np.max(calibrated_probs, axis=1)  # 标定后的概率的最大值

    # 打印 10 个样本的原始概率和标定后的概率（选择原始概率在 0.45 到 0.55 之间的样本）
    indices = np.where((original_max_probs >= 0.45) & (original_max_probs <= 0.55))[0]
    selected_indices = np.random.choice(indices, 10, replace=False)

    for i, idx in enumerate(selected_indices):
        print(
            f"Sample {idx+1}: Original Max Probability: {original_max_probs[idx]:.4f}, Calibrated Max Probability: {calibrated_max_probs[idx]:.4f}"
        )
