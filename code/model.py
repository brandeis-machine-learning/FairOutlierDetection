"""
Model skeleton for DCFOD

Date: 11/2020
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class DCFOD(nn.Module):
    """
    DCFOD consists of a encoder, decoder, discriminator, and cluster centroid layer
    """

    def __init__(self, input_size, num_classes, num_features, num_attributes, cuda):
        super(DCFOD, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_attributes = num_attributes
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, num_features)
        ).to(cuda)

        self.discriminator = nn.Sequential(
            nn.Linear(num_features, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, num_attributes)
        ).to(cuda)

        self.decoder = nn.Sequential(
            nn.Linear(num_features, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_size)
        ).to(cuda)

        self.clusterCenter = nn.Parameter(torch.zeros(num_classes, num_features).to(cuda))

        self.alpha = 1.0
        self.clusteringMode = False
        self.validateMode = False

        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight).to(cuda)

    def setClusteringMode(self, mode):
        self.clusteringMode = mode

    def setValidateMode(self, mode):
        self.validateMode = mode

    def getDistanceToClusters(self, x):
        """
        obtain the distance to cluster centroids for each instance
        Args:
            x: sample on the embedded space

        Returns: square of the euclidean distance, and the euclidean distance

        """
        xe = torch.unsqueeze(x, 1) - self.clusterCenter
        dist_to_centers = torch.sum(torch.mul(xe, xe), 2)
        euclidean_dist = torch.sqrt(dist_to_centers)

        return dist_to_centers, euclidean_dist

    def forward(self, x):
        # -----feature embedding----- #
        x = x.view(-1, self.input_size)
        x_e = self.encoder(x)

        # -----if only wants to initialize cluster centroids
        # or validate the performance on the whole dataset, return x_e----- #
        if self.clusteringMode or self.validateMode:
            return x_e

        # -----else the discriminator predicts the subgroup assignment for each instance----- #
        reversed_x_e = GradientReversalLayer.apply(x_e)
        x_sa = self.discriminator(reversed_x_e)

        # -----if in training, return the embedded x, decoded x, and subgroup discrimination----- #
        x_de = self.decoder(x_e)
        x_de = x_de.view(-1, 1, self.input_size)

        return x_e, x_de, x_sa


class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
