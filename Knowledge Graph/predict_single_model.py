# 在所有导入之前
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 设置随机种子（保持一致性）
def set_random_seeds(seed):
    """设置所有可能的随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
    # GPU内存配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置警告: {e}")

SEED = 20
set_random_seeds(SEED)

# ============================================================================
# 单模型预测函数
# ============================================================================
def predict_with_single_model(combined_matrix, model_path):
    """
    使用单个模型进行预测
    
    参数:
        combined_matrix: numpy数组，形状为 (n_samples, 21, 768)
        model_path: 模型文件的完整路径，例如 'path/to/best_model_fold_0.h5'
    
    返回:
        predictions: 预测结果数组
    """
    print("="*60)
    print("使用单个模型进行预测")
    print("="*60)
    
    # 检查输入数据形状
    print(f"\n输入数据形状: {combined_matrix.shape}")
    if len(combined_matrix.shape) != 3 or combined_matrix.shape[1:] != (21, 768):
        raise ValueError(f"输入数据形状应为 (n_samples, 21, 768)，当前为 {combined_matrix.shape}")
    
    # 调整数据形状以匹配模型输入 (n_samples, 1, 21, 768)
    X_pred = np.expand_dims(combined_matrix, axis=1)
    print(f"调整后形状: {X_pred.shape}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"\n加载模型: {model_path}")
    
    # 加载模型
    model = load_model(model_path)
    
    # 预测
    print("\n开始预测...")
    predictions = model.predict(X_pred, verbose=0).reshape(-1)
    
    print("\n" + "="*60)
    print("预测结果:")
    print("="*60)
    for i, pred in enumerate(predictions):
        print(f"样本 {i+1}: HV = {pred:.4f}")
    
    # 清理内存
    del model
    tf.keras.backend.clear_session()
    
    return predictions

