import paddle
import paddle.nn.functional as F

from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor


class LeNet(paddle.nn.Layer):

    def __init__(self):
        #
        super(LeNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5, out_features=120)

        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        #
        # x = x.reshape((-1, 1, 28, 28))
        #
        x = self.conv1(x)  # 图像：Tensor(64, 1, 28, 28) -> Tensor(64, 6, 28, 28)

        x = F.relu(x)

        x = self.max_pool1(x)  # 图像：Tensor(64, 6, 28, 28) -> Tensor(64, 6, 14, 14)

        x = F.relu(x)

        x = self.conv2(x)  # 图像：Tensor(64, 6, 14, 14) -> Tensor(64, 16, 10, 10)

        x = self.max_pool2(x)

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)  # Tensor(64, 16, 10, 10) -> Tensor(64, 400)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return x  # Tensor(64, 10)


def train(model, optim):
    #
    model.train()

    epochs = 2

    for epoch in range(epochs):

        for batch_id, data in enumerate(train_loader()):

            x_data = data[0]  # 图像：Tensor(64, 1, 28, 28)
            y_data = data[1]  # 标签：Tensor(64, 1)

            predicts = model(x_data)

            loss = F.cross_entropy(predicts, y_data)

            # calc loss
            acc = paddle.metric.accuracy(predicts, y_data)

            loss.backward()

            if batch_id % 300 == 0:
                #
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                    epoch,
                    batch_id,
                    loss.numpy(),
                    acc.numpy()
                ))

            optim.step()

            optim.clear_grad()


if __name__ == '__main__':
    #
    # paddle version
    #
    print(paddle.__version__)

    # prepare datasets
    train_dataset = MNIST(mode='train', transform=ToTensor())
    test_dataset = MNIST(mode='test', transform=ToTensor())

    # load dataset
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # build network
    model = LeNet()

    # prepare optimizer
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    # train network
    train(model, optim)

    # save training format model
    paddle.save(model.state_dict(), 'lenet.pdparams')
    paddle.save(optim.state_dict(), "lenet.pdopt")

    # load training format model
    model_state_dict = paddle.load('lenet.pdparams')
    opt_state_dict = paddle.load('lenet.pdopt')

    model.set_state_dict(model_state_dict)
    optim.set_state_dict(opt_state_dict)

    # save inferencing format model
    net = to_static(model, input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])

    paddle.jit.save(net, 'inference_model/lenet')
