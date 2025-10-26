# app/predict.py 最终修复版（解决StandardScaler未定义+模型找不到+导入问题）
# ===================== 第一步：导入依赖（新增StandardScaler导入）=====================
import os
import sys
import numpy as np
import joblib
from dotenv import load_dotenv
import mlflow
from mlflow.pyfunc import PyFuncModel
from sklearn.preprocessing import StandardScaler  # 关键：新增导入，解决NameError

# ===================== 第二步：强制添加根目录到搜索路径=====================
current_script_path = os.path.abspath(__file__)
app_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(app_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ 根目录已加入搜索路径：{project_root}")

# ===================== 第三步：配置MLflow+常量定义=====================
load_dotenv()
mlflow.set_tracking_uri(None)  # 本地模式
FASHION_MNIST_MODEL_NAME = "Fashion-MNIST-Logistic-Regression-Model"
SCALER_ARTIFACT_PATH = "preprocessing/fashion_mnist_scaler.pkl"

# ===================== 第四步：加载模型和标准化器=====================
def load_trained_model_and_scaler(
    model_name: str = FASHION_MNIST_MODEL_NAME,
    model_stage: str = "Latest"
) -> tuple[PyFuncModel, StandardScaler]:  # 现在StandardScaler已定义
    """从MLflow加载模型和对应的标准化器"""
    try:
        print(f"🔍 正在加载MLflow模型：{model_name}:{model_stage}")
        model_uri = f"models:/{model_name}/{model_stage.lower()}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 加载标准化器
        client = mlflow.tracking.MlflowClient()
        latest_model_version = client.get_latest_versions(model_name)[0]
        run_id = latest_model_version.run_id
        scaler_local_path = client.download_artifacts(run_id=run_id, path=SCALER_ARTIFACT_PATH)
        scaler = joblib.load(scaler_local_path)
        
        print(f"✅ 模型和标准化器加载完成（run ID：{run_id[:8]}...）")
        return model, scaler
    
    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(
            f"❌ 模型加载失败！请先运行 app/model.py 训练Fashion MNIST模型。\n"
            f"错误原因：{str(e)}"
        ) from e

# ===================== 第五步：预测函数=====================
def predict_fashion_mnist(
    model: PyFuncModel,
    scaler: StandardScaler,
    input_features: list[float]
) -> tuple[int, str]:
    """预测Fashion MNIST类别（输入784维特征，输出类别索引+名称）"""
    fashion_mnist_classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # 验证输入维度
    if len(input_features) != 784:
        raise ValueError(
            f"❌ 输入特征维度错误！需784个数值（28x28图像展平），实际输入{len(input_features)}个"
        )
    
    # 标准化输入+预测
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    class_index = model.predict(input_scaled)[0].astype(int)
    class_name = fashion_mnist_classes[class_index]
    
    return class_index, class_name

# ===================== 第六步：本地测试=====================
if __name__ == "__main__":
    try:
        # 加载模型和标准化器
        model, scaler = load_trained_model_and_scaler()
        
        # 测试样本：模拟Fashion MNIST标准化后的784维特征（以"T-shirt/top"为例）
        test_sample = np.random.normal(loc=0, scale=1, size=784).tolist()
        
        # 预测并输出结果
        class_index, class_name = predict_fashion_mnist(model, scaler, test_sample)
        print("\n=== 预测结果 ===")
        print(f"输入特征维度：{len(test_sample)}（符合784维要求）")
        print(f"预测类别索引：{class_index}")
        print(f"预测类别名称：{class_name}")
        
    except Exception as e:
        print(f"\n❌ 预测失败：{str(e)}")