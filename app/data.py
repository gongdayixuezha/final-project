# app/data.py 完整代码（仅加载Fashion MNIST，无旧数据残留）
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_idx_file(file_path: str) -> np.ndarray:
    """专用：读取Fashion MNIST的idx3-ubyte/idx1-ubyte格式文件"""
    import struct
    # 验证文件是否存在，给出明确错误提示
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Fashion MNIST文件不存在！请确认路径：\n{file_path}\n"
            "正确路径应为：E:\\VSproject\\final-project\\data\\raw\\train-images-idx3-ubyte\n"
            "提示：需将Fashion MNIST的4个idx文件解压到data/raw/目录（无.gz后缀）"
        )
    with open(file_path, "rb") as f:
        magic_number, num_items = struct.unpack(">II", f.read(8))
        if magic_number == 2051:  # 图像文件（28x28=784特征）
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows * cols)
        elif magic_number == 2049:  # 标签文件（10分类）
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"❌ 不是Fashion MNIST文件！魔法数：{magic_number}")
    return data


def load_local_fashion_mnist(scale_data: bool = True) -> tuple:
    """
    加载本地Fashion MNIST数据集（10分类），返回标准化后的数据
    返回值：X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    load_dotenv()  # 加载.env配置（若未配置，默认使用data/目录）
    # 定义数据路径（强制对应项目结构）
    data_root = os.getenv("LOCAL_DATA_ROOT", "data/")  # 根目录下的data文件夹
    train_img_path = os.path.join(data_root, "raw", "train-images-idx3-ubyte")
    train_lab_path = os.path.join(data_root, "raw", "train-labels-idx1-ubyte")
    test_img_path = os.path.join(data_root, "raw", "t10k-images-idx3-ubyte")
    test_lab_path = os.path.join(data_root, "raw", "t10k-labels-idx1-ubyte")
    # 读取并验证数据维度（Fashion MNIST固定维度，确保不是旧数据）
    print("🔍 正在加载Fashion MNIST数据...")
    X_train = load_idx_file(train_img_path)
    y_train = load_idx_file(train_lab_path)
    X_test = load_idx_file(test_img_path)
    y_test = load_idx_file(test_lab_path)
    # 强制验证数据正确性（避免加载鸢尾花等旧数据）
    assert X_train.shape == (60000, 784), f"❌ 数据维度错误！训练集应为(60000,784)，实际为{X_train.shape}"
    assert y_train.shape == (60000,), f"❌ 标签维度错误！训练标签应为(60000,)，实际为{y_train.shape}"
    assert len(set(y_train)) == 10, f"❌ 分类数错误！Fashion MNIST是10分类，实际为{len(set(y_train))}"
    # 标准化（解决模型收敛问题，必须执行）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if scale_data else X_train
    X_test_scaled = scaler.transform(X_test) if scale_data else X_test
    # 打印加载结果（方便用户验证）
    print(f"✅ Fashion MNIST加载完成：")
    print(f"  - 训练集：{X_train_scaled.shape} | 训练标签：{y_train.shape}")
    print(f"  - 测试集：{X_test_scaled.shape} | 测试标签：{y_test.shape}")
    print(f"  - 标准化生效：训练集均值={X_train_scaled.mean():.4f}（接近0）")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 单独运行data.py时验证数据（用户可执行此文件确认数据正确性）
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist()
        print("\n📊 数据验证通过！可正常用于模型训练")
    except Exception as e:
        print(f"\n❌ 数据加载失败：{str(e)}")