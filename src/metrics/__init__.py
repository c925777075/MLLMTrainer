# 想计算什么metric自己写就行了

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()  # 确保预测值是平坦的
    labels = labels.flatten()
    mse = mean_squared_error(labels, predictions)
    return {'mse': mse}