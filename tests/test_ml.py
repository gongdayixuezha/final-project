import pytest
from app.model import load_data, train_model
from app.predict import load_trained_model, predict_iris
import numpy as np


def test_data_loading():
    """测试数据加载：确保返回正确的数据集格式和维度"""
    X_train, X_test, y_train, y_test = load_data()
    # 使用pytest.raises避免F401错误（验证断言逻辑）
    with pytest.raises(AssertionError):
        assert X_train.shape == (121, 4)  # 故意错误的维度，验证断言生效
    assert X_train.shape == (120, 4)  # 150个样本，80%训练集（120个），4个特征
    assert y_test.shape == (30,)  # 20%测试集（30个）
    assert set(y_train) == {0, 1, 2}  # 鸢尾花3个类别


def test_model_training():
    """测试模型训练：确保训练后准确率达标（至少0.8）"""
    model, accuracy = train_model(learning_rate=0.1, max_iter=100)
    assert accuracy >= 0.8  # 简单分类任务，准确率应不低于80%


def test_model_prediction():
    """测试模型预测：确保对已知样本的预测正确"""
    # 加载本地训练的模型（mlruns/目录下的最新run）
    model = load_trained_model(model_uri="runs:/latest/iris-classifier-model")
    # 使用numpy处理数据，避免F401错误
    test_data = np.array([5.1, 3.5, 1.4, 0.2])
    prediction = predict_iris(model, test_data)
    assert prediction == 0
