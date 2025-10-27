# app/model.py 完整代码（解决ModuleNotFoundError+收敛警告+模型名混淆）
# ===================== 第一步：强制添加根目录到搜索路径（解决导入问题）=====================
import os
import sys

# 1. 计算项目根目录（无论脚本怎么运行，都能找到final-project目录）
current_script_path = os.path.abspath(
    __file__
)  # 当前model.py的绝对路径（如：e:/VSproject/final-project/app/model.py）
app_dir = os.path.dirname(current_script_path)  # 上级目录：app/
project_root = os.path.dirname(app_dir)  # 上上级目录：final-project（根目录）

# 2. 强制将根目录加入Python模块搜索路径（优先搜索，避免导入失败）
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ 根目录已加入模块搜索路径：{project_root}")
else:
    print(f"✅ 根目录已在搜索路径中：{project_root}")

# ===================== 第二步：导入依赖库（必须在路径配置之后）=====================
from dotenv import load_dotenv
import mlflow
import joblib
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 导入Fashion MNIST数据加载函数（此时app包已能正常识别）
from app.data import load_local_fashion_mnist

# ===================== 第三步：配置MLflow（适配Fashion MNIST）=====================
load_dotenv()  # 加载.env（若配置了MLflow地址，优先使用；无则默认本地模式）
# 强制设置Fashion MNIST专属实验（避免与鸢尾花实验混淆）
mlflow.set_experiment("Fashion-MNIST-Logistic-Regression-Experiment")
print(f"✅ MLflow实验已配置：Fashion-MNIST-Logistic-Regression-Experiment")


# ===================== 第四步：模型训练函数（解决收敛问题）=====================
def train_model(learning_rate: float = 0.1, max_iter: int = 1000) -> tuple:
    """
    训练适配Fashion MNIST的逻辑回归模型（解决收敛警告）
    参数：
        learning_rate: 学习率（对应正则化强度C=1/learning_rate）
        max_iter: 迭代次数（设为1000确保solver收敛）
    返回：
        model: 训练好的逻辑回归模型
        test_accuracy: 测试集准确率（正常范围：0.89-0.91）
    """
    # 1. 加载Fashion MNIST数据（强制使用标准化后的数据）
    X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist(scale_data=True)
    # 2. 启动MLflow Run（记录实验参数和结果）
    with mlflow.start_run(
        run_name=f"LR-lr{learning_rate}-iter{max_iter}-saga"  # 包含求解器名，方便区分
    ) as run:
        # 3. 记录关键参数（确保可追溯）
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", "saga")  # 关键：用saga求解器，适配多分类+大样本
        mlflow.log_param("max_iter", max_iter)  # 关键：提高迭代次数到1000
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("regularization_strength_C", 1 / learning_rate)
        mlflow.log_param("multi_class_strategy", "ovr")  # 适配10分类
        mlflow.log_param("dataset", "Fashion MNIST (60000 train / 10000 test)")
        mlflow.log_param("data_preprocessing", "StandardScaler (mean≈0, std=1)")
        # 4. 初始化模型（参数强制适配多分类和收敛）
        model = LogisticRegression(
            C=1 / learning_rate,  # 正则化强度（与学习率成反比）
            solver="saga",  # 解决收敛：适合多分类+大样本
            max_iter=max_iter,  # 解决收敛：足够的迭代次数
            multi_class="ovr",  # 10分类策略（One-vs-Rest）
            random_state=42,  # 固定随机种子，结果可复现
            n_jobs=-1,  # 用所有CPU核心加速训练
        )
        # 5. 训练模型（无收敛警告）
        print(f"📌 开始训练模型：lr={learning_rate}, max_iter={max_iter}, solver=saga")
        model.fit(X_train, y_train)
        # 6. 评估模型（计算准确率）
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = model.score(X_train, y_train)
        print(
            f"✅ 训练完成：训练准确率={train_accuracy:.4f}, 测试准确率={test_accuracy:.4f}"
        )
        # 7. 记录MLflow指标（方便后续对比）
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_iterations_used", model.n_iter_[0])  # 实际迭代次数
        # 8. 记录模型和预处理 artifacts（方便后续预测）
        # 8.1 记录标准化器（预测时需用相同的scaler）
        joblib.dump(scaler, "fashion_mnist_scaler.pkl")
        mlflow.log_artifact("fashion_mnist_scaler.pkl", artifact_path="preprocessing")
        os.remove("fashion_mnist_scaler.pkl")  # 清理本地临时文件

        # 8.2 记录模型（注册到MLflow，名称不含鸢尾花）
        signature = infer_signature(
            X_train, model.predict(X_train)
        )  # 自动推断输入输出格式
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fashion-mnist-lr-model",
            signature=signature,
            registered_model_name="Fashion-MNIST-Logistic-Regression-Model",  # 新模型名，无旧残留
        )
        print(f"✅ 模型已注册到MLflow：Fashion-MNIST-Logistic-Regression-Model")
        # 9. 记录数据加载脚本（确保可复现）
        mlflow.log_artifact("app/data.py", artifact_path="scripts")
        mlflow.log_artifact("app/model.py", artifact_path="scripts")
    return model, test_accuracy


# ===================== 第五步：本地测试（验证代码运行）=====================
if __name__ == "__main__":
    print("\n=== 开始Fashion MNIST模型训练（本地测试）===")
    # 测试2组超参数（覆盖常见学习率）
    experiment_1 = {"learning_rate": 0.1, "max_iter": 1000}
    experiment_2 = {"learning_rate": 0.01, "max_iter": 1000}
    # 运行实验1
    print(f"\n📊 实验1：{experiment_1}")
    model1, acc1 = train_model(**experiment_1)
    # 运行实验2
    print(f"\n📊 实验2：{experiment_2}")
    model2, acc2 = train_model(**experiment_2)
    # 输出最终结果（验证是否正常）
    print("\n=== 训练结果汇总 ===")
    print(f"实验1（lr=0.1, iter=1000）测试准确率：{acc1:.4f}（正常范围：0.89-0.91）")
    print(f"实验2（lr=0.01, iter=1000）测试准确率：{acc2:.4f}（正常范围：0.89-0.91）")
    print(f"\n✅ 查看MLflow实验详情：")
    print(f"1. 终端执行命令：mlflow ui")
    print(f"2. 浏览器访问：http://localhost:5000")
    print(f"3. 实验路径：{os.path.abspath('mlruns/')}")
    # 强制验证准确率（避免加载旧数据）
    assert (
        acc1 < 0.95
    ), "❌ 警告：准确率异常高（>0.95），可能加载了鸢尾花数据！请检查data.py"
    assert (
        acc2 < 0.95
    ), "❌ 警告：准确率异常高（>0.95），可能加载了鸢尾花数据！请检查data.py"
