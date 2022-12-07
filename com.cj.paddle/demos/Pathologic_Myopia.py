import paddle
import cv2
import numpy as np
import os
import random
from paddle.vision.models import resnet50, vgg16, LeNet

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放到 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H ,W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode='train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)

    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []

        for name in filenames:
            img_path = os.path.join(datadir, name)
            img = cv2.imread(img_path)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近视，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                label = 1
            else:
                raise ('NOT EXCEPTED FILE NAME')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


# 定义训练过程
def train_pm(model, optimizer):
    use_gpu = False
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    print('start training ... ')
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=10, mode='train')
    for epoch in range(EPOCH_NUM):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            imgs = paddle.to_tensor(x_data)
            labels = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            predicts = model(imgs)
            loss = paddle.nn.functional.binary_cross_entropy_with_logits(predicts, labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 20 == 0:
                print('epoch: {}, batch_id: {}, loss is: {:.4f}'.format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 计算准确率
        model.eval()
        accuracies = []
        losses = []
        eval_loader = data_loader(DATADIR, batch_size=10, mode='eval')
        for batch_id, data in enumerate(eval_loader()):
            x_data, y_data = data
            imgs = paddle.to_tensor(x_data)
            labels = paddle.to_tensor(y_data)
            # 进行模型前向计算，得到预测值
            logits = model(imgs)
            # 计算sigmoid后的预测概率，进行loss计算
            loss = paddle.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            predicts2 = paddle.nn.functional.sigmoid(logits)
            predicts3 = predicts2 * (-1.0) + 1.0
            pred = paddle.concat([predicts3, predicts2], axis=1)
            acc = paddle.metric.accuracy(pred, paddle.cast(labels, dtype='int64'))

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print('[validation] accuracy/loss: {:.4f}/{:.4f}'.format(np.mean(accuracies), np.mean(losses)))

        paddle.save(model.state_dict(), '../output/pathologic_myopia/vgg/palm.pdparams')
        paddle.save(optimizer.state_dict(), '../output/pathologic_myopia/vgg/palm.pdopt')


# 定义评估过程
def evaluation(model, params_file_path):
    # 开启0号GPU预估
    use_gpu = False
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    print('start evaluation ......')

    # 加载参数模型
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)

    # 开启评估
    model.eval()
    eval_loader = data_loader(DATADIR, batch_size=10, mode='eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        label_64 = paddle.to_tensor(y_data.astype(np.int64))
        # 计算预测和精度(label_64作用???算出来的精确度???)
        prediction, acc = model(img, label_64)
        # 计算损失函数值
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(prediction, label)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 求平均精度
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={:.4f}, acc={:.4f}'.format(avg_loss_val_mean, acc_val_mean))


# 定义模型
class MyLeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(MyLeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2池化层
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=6, kernel_size=5)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 创建第三个卷积层
        self.conv3 = paddle.nn.Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 创建全连接层，第一个全连接层的输出神经元个数为64
        features_num = pow(((224 - 4) // 2 - 4) // 2 - 3, 2) * 120
        self.fc1 = paddle.nn.Linear(in_features=features_num, out_features=64)
        self.fc2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    # 网络前向计算过程
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = paddle.nn.functional.sigmoid(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = paddle.nn.functional.sigmoid(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = paddle.nn.functional.sigmoid(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = paddle.nn.functional.sigmoid(x)
        x = self.fc2(x)
        # 如果输入了Label，那么计算acc
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 优化模型
class MyAlexNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(MyAlexNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = paddle.nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = paddle.nn.Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = paddle.nn.Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.fc1 = paddle.nn.Linear(in_features=12544, out_features=4096)
        self.drop_ratio1 = 0.5
        self.drop1 = paddle.nn.Dropout(self.drop_ratio1)
        self.fc2 = paddle.nn.Linear(in_features=4096, out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = paddle.nn.Dropout(self.drop_ratio2)
        self.fc3 = paddle.nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv4(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv5(x)
        x = paddle.nn.functional.relu(x)
        x = self.max_pool5(x)
        # 全连接层
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        # 在全连接之后使用dropout抑制过拟合
        x = self.drop1(x)
        x = self.fc2(x)
        x = paddle.nn.functional.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)

        # 如果输入了Label，那么计算acc
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 优化模型
class VGG(paddle.nn.Layer):
    def __init__(self, num_classes=None):
        super(VGG, self).__init__()

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个block，包含两个卷积
        self.conv1_1 = paddle.nn.Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3,
                                        padding=1, stride=1)
        self.conv1_2 = paddle.nn.Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3,
                                        padding=1, stride=1)
        # 定义第二个block，包含两个卷积
        self.conv2_1 = paddle.nn.Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3,
                                        padding=1, stride=1)
        self.conv2_2 = paddle.nn.Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3,
                                        padding=1, stride=1)
        # 定义第三个block，包含三个卷积
        self.conv3_1 = paddle.nn.Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3,
                                        padding=1, stride=1)
        self.conv3_2 = paddle.nn.Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3,
                                        padding=1, stride=1)
        self.conv3_3 = paddle.nn.Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3,
                                        padding=1, stride=1)
        # 定义第四个block，包含三个卷积
        self.conv4_1 = paddle.nn.Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3,
                                        padding=1, stride=1)
        self.conv4_2 = paddle.nn.Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3,
                                        padding=1, stride=1)
        self.conv4_3 = paddle.nn.Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3,
                                        padding=1, stride=1)
        # 定义第五个block，包含三个卷积
        self.conv5_1 = paddle.nn.Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3,
                                        padding=1, stride=1)
        self.conv5_2 = paddle.nn.Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3,
                                        padding=1, stride=1)
        self.conv5_3 = paddle.nn.Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3,
                                        padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成一个线性结构 (fc + relu)
        # 当输入为244x244时，经过五个卷积块和池化层后，特征维度变为[512x7x7]
        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential将全连接层和relu组成一个线性结构(fc + relu)
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())
        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, 1)

        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x, label=None):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)

        # 如果输入了Label，那么计算acc
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 优化模型
# 定义Inception块
class Inception(paddle.nn.Layer):
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码
        :param c0:
        :param c1:图(b)中第一条支路1x1卷积的输出通道数,数据类型是整数
        :param c2:图(b)中第二条支路卷积的输出通道数,数据类型是tuple或list,其中c2[0]是1x1卷积的输出通道数,c2[1]是3x3
        :param c3:图(b)中第三条支路卷积的输出通道数,数据类型是tuple或list,其中c3[0]是1x1卷积的输出通道数,c3[1]是3x3
        :param c4:图(b)中第一条支路1x1卷积的输出通道数,数据类型是整数
        :param kwargs:
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = paddle.nn.Conv2D(in_channels=c0, out_channels=c1, kernel_size=1, stride=1)

        self.p2_1 = paddle.nn.Conv2D(in_channels=c0, out_channels=c2[0], kernel_size=1, stride=1)
        self.p2_2 = paddle.nn.Conv2D(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1)

        self.p3_1 = paddle.nn.Conv2D(in_channels=c0, out_channels=c3[0], kernel_size=1, stride=1)
        self.p3_2 = paddle.nn.Conv2D(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1)

        self.p4_1 = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = paddle.nn.Conv2D(in_channels=c0, out_channels=c4, kernel_size=1, stride=1)

        # # 新加一层batchnorm稳定收敛
        # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

    def forward(self, x):
        # 支路1只包含一个 1x1卷积
        p1 = paddle.nn.functional.relu(self.p1_1(x))
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = paddle.nn.functional.relu(self.p2_2(paddle.nn.functional.relu(self.p2_1(x))))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = paddle.nn.functional.relu(self.p3_2(paddle.nn.functional.relu(self.p3_1(x))))
        # 支路4包含 最大池化 + 1x1卷积
        p4 = paddle.nn.functional.relu(self.p4_2(paddle.nn.functional.relu(self.p4_1(x))))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return paddle.concat([p1, p2, p3, p4], axis=1)
        # return self.batchnorm()


class GoogleNet(paddle.nn.Layer):
    def __init__(self, num_classes=None):
        super(GoogleNet, self).__init__()
        # GoogleLeNet包含5个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=1)
        # 3x3最大池化
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第二个模块包含2个卷积层
        self.conv2_1 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = paddle.nn.Conv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        # 3x3最大池化
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.max_pool3 = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3x3 最大池化
        self.max_pool4 = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，用的是global_pooling，不需要pool_stride
        self.pool5 = paddle.nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = paddle.nn.Linear(in_features=1024, out_features=1)

    def forward(self, x, label=None):
        x = self.max_pool1(paddle.nn.functional.relu(self.conv1(x)))
        x = self.max_pool2(paddle.nn.functional.relu(self.conv2_2(paddle.nn.functional.relu(self.conv2_1(x)))))
        x = self.max_pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.max_pool4(self.block4_5(self.block4_4(x)))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        # 如果输入了Label，那么计算acc
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一体化
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        '''
        :param num_channels:卷积层的输入通道数
        :param num_filters: 卷积层的输出通道数
        :param filter_size:
        :param stride: 卷积层的步幅
        :param groups: 分组卷积的数组，默认groups=1不使用分组卷积
        :param act:
        '''
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = paddle.nn.Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=filter_size,
                                      stride=stride, padding=(filter_size - 1) // 2, groups=groups, bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = paddle.nn.functional.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = paddle.nn.functional.relu(x=y)

        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=num_filters, filter_size=1, act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters, filter_size=3, stride=stride,
                                 act='relu')
        # 创建第三个卷积1x1，但是输出通道数乘以4
        self.conv2 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters * 4, filter_size=1, act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels, num_filters=num_filters * 4, filter_size=1,
                                     stride=stride)

        self.shortcut = shortcut
        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = paddle.nn.functional.relu(y)

        return y


# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        '''

        :param layers:网络层数，可以是50,101或者152
        :param class_dim: 分类标签的类别数
        '''
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, 'supporrted layers are {} but input layer is {}'.format(supported_layers,
                                                                                                   layers)

        if layers == 50:
            # ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNET152包含多个模块，其中第2到第5个模块分别包含3、8、36、3
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.max_pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # ResNet的第二个到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(num_channels=num_channels,
                                    num_filters=num_filters[
                                        block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    shortcut=shortcut)
                )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，卷积特征的维度是[B, 2048, 1, 1]，故最后一层全连接的输入维度是2048
        self.out = paddle.nn.Linear(in_features=2048,
                                    out_features=class_dim,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.Uniform(-stdv, stdv))
                                    )

    def forward(self, inputs, label=None):
        y = self.conv(inputs)
        y = self.max_pool(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)

        # 如果输入了Label，那么计算acc
        if label is not None:
            acc = paddle.metric.accuracy(input=inputs, label=label)
            return y, acc
        else:
            return y


# 查看数据形状
DATADIR = '../dataset/pathologic_myopia'
# 设置迭代轮数
EPOCH_NUM = 5
# 创建模型
# model = ResNet()
model = resnet50(pretrained=False, num_classes=1)
# 启动训练过程
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
train_pm(model=model, optimizer=opt)
evaluation(model, params_file_path='../output/pathologic_myopia/vgg/palm.pdparams')
