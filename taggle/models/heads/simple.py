from torch import nn

from ..common import AdaptiveConcatPool2d, ArcMarginProduct, Flatten, GeM


class SimpleHead(nn.Module):
    __name__ = 'SimpleHead'

    def __init__(self,
                 encoder_channels,
                 p=0.2,
                 metric_branch=False,
                 last_activation=None,
                 extract_feature=False,
                 pooling="ACP",
                 num_class=10):
        super(SimpleHead, self).__init__()
        if pooling == "GeM":
            self.pooling = GeM()
            dense_input = encoder_channels[0]
        elif pooling == "ACP":
            self.pooling = AdaptiveConcatPool2d()
            dense_input = encoder_channels[0] * 2
        elif pooling == "GMP":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
            dense_input = encoder_channels[0]
        elif pooling == "GAP":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            dense_input = encoder_channels[0]
        self.flatten = Flatten()

        self.dense_layers = nn.Sequential(
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
        x = self.flatten(self.pooling(feats[0]))
        logits = self.dense_layers(x)
        logits = self.last_layer(logits)
        if self.activation:
            logits = self.activation(logits)
        return logits
