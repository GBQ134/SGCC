import time
import random
import joblib
import argparse
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# 全局配置
feat_to_use = []  # 特征索引列表
class_index = 6  # 类别索引
debug = True


def load_features_and_class(filepath):
    """加载特征和类别索引配置"""
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]
            elif line_index == 1:
                global class_index
                class_index = int(tokens[0])


def read_data(filepath):
    """加载点云数据"""
    X, Y = [], []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            if 'nan' not in tokens:
                X.append([float(t) for t_index, t in enumerate(tokens) if t_index != class_index])
                Y.append(int(float(tokens[class_index])))

    Xarray = np.asarray(X, dtype=np.float32)
    Yarray = np.asarray(Y, dtype=np.float32)
    point_setf = np.zeros((len(X), 6))

    point_setf[:, 0:3] = pc_normalize(Xarray[:, 0:3])
    point_setf[:, 3:6] = Xarray[:, 3:6]
    return point_setf, Yarray


def pc_normalize(pc):
    """点云归一化"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def train_model(X_train, Y_train, params, n_jobs):
    """训练随机森林模型"""
    model = RandomForestClassifier(
        n_estimators=params['ne'],
        min_samples_leaf=params['msl'],
        min_samples_split=params['mss'],
        max_depth=params['md'],
        criterion='entropy',
        oob_score=True,
        n_jobs=n_jobs,
        random_state=0
    )
    model.fit(X_train[:, feat_to_use], Y_train)
    return model


def bayesian_optimization(X, y, n_jobs):
    """贝叶斯优化核心函数"""
    # 定义参数空间
    space = [
        Integer(25, 30, name='md'),  # max_depth
        Integer(500, 800, name='ne'),  # n_estimators
        Integer(1, 3, name='msl'),  # min_samples_leaf
        Integer(2, 5, name='mss')  # min_samples_split
    ]

    # 目标函数
    def objective(params):
        model = RandomForestClassifier(
            max_depth=params[0],
            n_estimators=params[1],
            min_samples_leaf=params[2],
            min_samples_split=params[3],
            criterion='entropy',
            n_jobs=n_jobs,
            random_state=0
        )

        # 五折交叉验证
        scores = cross_val_score(
            model, X[:, feat_to_use], y,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        return -np.mean(scores)  # 最小化目标

    # 执行贝叶斯优化
    res = gp_minimize(
        objective,
        space,
        n_calls=100,  # 总迭代次数
        n_initial_points=20,  # 初始随机采样
        acq_func='EI',  # 期望提升策略
        random_state=42,
        verbose=True
    )

    # 返回最优参数组合
    return {
        'md': res.x[0],
        'ne': res.x[1],
        'msl': res.x[2],
        'mss': res.x[3]
    }


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='随机森林岩性识别')
    parser.add_argument('--features_filepath', default="featurefile.txt", help='特征配置文件路径')
    parser.add_argument('--training_filepath', default="800f.txt", help='训练数据路径')
    parser.add_argument('--test_filepath', default="800f.txt", help='测试数据路径')
    parser.add_argument('--n_jobs', default=16, type=int, help='并行线程数')
    parser.add_argument('--output_name', default="pred.txt", help='输出文件名')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载数据
    print("Loading data...")
    load_features_and_class(args.features_filepath)
    X_train, Y_train = read_data(args.training_filepath)

    # 数据划分
    Xsize = X_train.shape[0]
    indices = np.random.permutation(Xsize)
    train_size = int(Xsize * 0.95)

    X_train = X_train[indices[:train_size]]
    Y_train = Y_train[indices[:train_size]]
    X_test = X_train[indices[train_size:]]
    Y_test = Y_train[indices[train_size:]]

    print(f"\t训练样本: {len(Y_train)}\n\t测试样本: {len(Y_test)}\n\t使用特征索引: {feat_to_use}")

    # 执行贝叶斯优化
    print("\nStarting Bayesian optimization...")
    start_opt = time.time()
    best_params = bayesian_optimization(X_train, Y_train, args.n_jobs)
    opt_time = time.time() - start_opt

    # 训练最终模型
    print("\nTraining final model...")
    start_train = time.time()
    final_model = train_model(X_train, Y_train, best_params, args.n_jobs)
    train_time = time.time() - start_train

    # 模型评估
    Y_pred = final_model.predict(X_test[:, feat_to_use])
    final_f1 = f1_score(Y_test, Y_pred, average='weighted')
    final_acc = accuracy_score(Y_test, Y_pred)

    # 输出结果
    print("\n=== 优化结果 ===")
    print(f"优化耗时: {opt_time:.1f}s")
    print(f"训练耗时: {train_time:.1f}s")
    print(f"最佳参数: {best_params}")
    print(f"验证集F1: {final_f1:.4f}")
    print(f"验证集准确率: {final_acc:.4f}")
    print("特征重要性:\n", final_model.feature_importances_)
    print("混淆矩阵:\n", confusion_matrix(Y_test, Y_pred))

    # 保存模型
    model_name = f"RF_md{best_params['md']}_ne{best_params['ne']}_f1{final_f1:.2f}.pkl"
    save_model(final_model, model_name)
    print(f"\n模型已保存为: {model_name}")


def save_model(model, filename):
    """保存训练好的模型"""
    with open(filename, 'wb') as f:
        joblib.dump(model, f)


if __name__ == '__main__':
    main()