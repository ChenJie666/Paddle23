# 现在面临这样一个任务：
# 乘坐出租车，起步价为10元，每行驶1公里，需要在支付每公里2元
# 当一个乘客坐完出租车后，车上的计价器需要算出来该乘客需要支付的乘车费用

def calculate_fee(distance_travelled):
    return 10 + 2 * distance_travelled


for x in [1, 3, 5, 9, 10, 20]:
    print(calculate_fee(x))

# 结果为
# 12
# 16
# 20
# 28
# 30
# 50

# 接下来，把问题稍微变换一下，现在知道乘客每次出行的公里数和支付的总费用
# 需要求解乘车的起步价和每公里的费用
import paddle

x_data = paddle.to_tensor([[1.0], [3.0], [5.0], [9.0], [10.0], [20.0]])
y_data = paddle.to_tensor([[12.0], [16.0], [20.0], [28.0], [30.0], [50.0]])

linear = paddle.nn.Linear(in_features=1, out_features=1)
w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()
print(w_before_opt, b_before_opt)  # 随机初始化的值

mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())

total_epoch = 10000
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict, y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_gradients()

    if i % 1000 == 0:
        print(i, loss.numpy())

print("finish training, loss = {}".format(loss.numpy()))

w_after_opt = linear.weight.numpy().item()
b_after_opt = linear.bias.numpy().item()
print(w_after_opt, b_after_opt)  # 最终的拟合值