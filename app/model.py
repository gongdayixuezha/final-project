# app/model.py 优化版（使用公共路径工具+完善类型提示）
# ===================== 第一步：导入公共工具（解决路径配置重复）=====================
from app.utils import add_project_root_to_path

# 初始化项目路径
project_root = add_project_root_to_path()

# ===================== 第二步：导入依赖库 =====================
import os
from dotenv import load_dotenv
import mlflow
import joblib
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple  # 新增类型提示

# 导入数据加载函数
from app.data import load_local_fashion_mnist

# ===================== 第三步：配置MLflow =====================
load_dotenv()
mlflow.set_experiment("Fashion-MNIST-Logistic-Regression-Experiment")
print(f"✅ MLflow实验已配置：Fashion-MNIST-Logistic-Regression-Experiment")


# ===================== 第四步：模型训练函数 =====================
def train_model(learning_rate: float = 0.1, max_iter: int = 1000) -> Tuple[LogisticRegression, float]:
    """
    训练适配Fashion MNIST的逻辑回归模型
    参数：
        learning_rate: 学习率（对应正则化强度C=1/learning_rate）
        max_iter: 迭代次数
    返回：
        model: 训练好的逻辑回归模型
        test_accuracy: 测试集准确率
    """
    # 加载数据（标准化后的数据）
    X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist(scale_data=True)
    
    # 启动MLflow Run
    with mlflow.start_run(
        run_name=f"LR-lr{learning_rate}-iter{max_iter}-saga"
    ) as run:
        # 记录参数
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", "saga")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("regularization_strength_C", 1 / learning_rate)
        mlflow.log_param("multi_class_strategy", "ovr")
        mlflow.log_param("dataset", "Fashion MNIST (60000 train / 10000 test)")
        mlflow.log_param("data_preprocessing", "StandardScaler (mean≈0, std=1)")
        
        # 初始化模型
        model = LogisticRegression(
            C=1 / learning_rate,
            solver="saga",
            max_iter=max_iter,
            multi_class="ovr",
            random_state=42,
            n_jobs=-1,
        )
        
        # 训练模型
        print(f"📌 开始训练模型：lr={learning_rate}, max_iter={max_iter}, solver=saga")
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = model.score(X_train, y_train)
        print(
            f"✅ 训练完成：训练准确率={train_accuracy:.4f}, 测试准确率={test_accuracy:.4f}"
        )
        
        # 记录指标
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_iterations_used", model.n_iter_[0])
        
        # 记录标准化器
        joblib.dump(scaler, "fashion_mnist_scaler.pkl")
        mlflow.log_artifact("fashion_mnist_scaler.pkl", artifact_path="preprocessing")
        os.remove("fashion_mnist_scaler.pkl")
        
        # 记录模型
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fashion-mnist-lr-model",
            signature=signature,
            registered_model_name="Fashion-MNIST-Logistic-Regression-Model",
        )
        print(f"✅ 模型已注册到MLflow：Fashion-MNIST-Logistic-Regression-Model")
        
        # 记录脚本
        mlflow.log_artifact("app/data.py", artifact_path="scripts")
        mlflow.log_artifact("app/model.py", artifact_path="scripts")
    
    return model, test_accuracy


# ===================== 第五步：本地测试 =====================
if __name__ == "__main__":
    print("\n=== 开始Fashion MNIST模型训练（本地测试）===")
    experiment_1 = {"learning_rate": 0.1, "max_iter": 1000}
    experiment_2 = {"learning_rate": 0.01, "max_iter": 1000}
    
    print(f"\n📊 实验1：{experiment_1}")
    model1, acc1 = train_model(** experiment_1)
    
    print(f"\n📊 实验2：{experiment_2}")
    model2, acc2 = train_model(**experiment_2)
    
    print("\n=== 训练结果汇总 ===")
    print(f"实验1（lr=0.1, iter=1000）测试准确率：{acc1:.4f}（正常范围：0.89-0.91）")
    print(f"实验2（lr=0.01, iter=1000）测试准确率：{acc2:.4f}（正常范围：0.89-0.91）")
    print(f"\n✅ 查看MLflow实验详情：")
    print(f"1. 终端执行命令：mlflow ui")
    print(f"2. 浏览器访问：http://localhost:5000")
    print(f"3. 实验路径：{os.path.abspath('mlruns/')}")
    
    # 验证准确率
    assert acc1 < 0.95, "❌ 警告：准确率异常高（>0.95），可能加载了错误数据！"
    assert acc2 < 0.95, "❌ 警告：准确率异常高（>0.95），可能加载了错误数据！"