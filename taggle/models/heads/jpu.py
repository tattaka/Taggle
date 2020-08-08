from torch import nn

from ..common import (
    ASPP,
    JPU,
    AdaptiveConcatPool2d,
    ArcMarginProduct,
    Flatten,
    GeM
)


class JPUHead(nn.Module):
    __name__ = 'JPUHead'

    def __init__(self,
                 encoder_channels,
                 mid_channel=512,
                 p=0.2,
                 pooling="ACP",
                 metric_branch=False,
                 extract_feature = False,
                 last_activation=None,
                 num_class=10):
        super(JPUHead, self).__init__()

        self.jpu = JPU([encoder_channels[0], encoder_channels[1],
                        encoder_channels[2]], mid_channel)
        self.aspp = ASPP(mid_channel * 4, mid_channel,
                         dilations=[1, (1, 4), (2, 8), (3, 12)])

        if pooling == "GeM":
            self.pooling = GeM()
            dense_input = mid_channel
        elif pooling == "ACP":
            self.pooling = AdaptiveConcatPool2d()
            dense_input = mid_channel * 2
        elif pooling == "GMP":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
            dense_input = mid_channel
        elif pooling == "GAP":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            dense_input = mid_channel

        self.flatten = Flatten()
        self.dense_layers = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(dense_input),
            nn.Dropout(p),
            nn.Linear(dense_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p))
        if metric_branch:
            self.last_layer = ArcMarginProduct(1024, num_class, extract_feature)
        else:
            self.last_layer = nn.Linear(1024, num_class)
        if last_activation is None:
            self.activation = last_activation
        elif last_activation == 'LogSoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif last_activation == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif last_activation == 'Sigmoid':
            self.activation = nn.Sigmoid(dim=1)
        else:
            raise ValueError(
                'Activation should be "LogSoftmax"/"Softmax"/"Sigmoid"/None')

    def forward(self, feats):
        x = self.jpu(feats[0], feats[1], feats[2])
        x = self.aspp(x)
        x = self.flatten(self.pooling(x))
        logits = self.dense_layers(x)
        logits = self.last_layer(logits)
        if self.activation:
            logits = self.activation(logits)

        return logits
