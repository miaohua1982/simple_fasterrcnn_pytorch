from torch import nn
import torch
from torch.nn import functional as F
from model.util.align_roi_pool import RoIAlign_C


def pyramid_roi_align(feature_maps, boxes, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7] for classifier, [14, 14] for mask header
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    # Assign each ROI to a level in the pyramid based on the ROI area.
    x1, y1, x2, y2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = torch.FloatTensor([float(image_shape[0]*image_shape[1])])
    if torch.cuda.is_available():
        image_area = image_area.cuda()
    roi_level = 4 + torch.log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)


    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix, as_tuple=False)[:,0]  # [:,0] change shape [n,1] to [n]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = torch.zeros(level_boxes.size()[0]).int()
        if torch.cuda.is_available():
            ind = ind.cuda()
        
        scale = feature_maps[i].shape[-1]/image_shape[-1]   # calc the scale from current feature map to input image
        # feature_maps[i]: [batch_size, channels, height, width]
        # level_boxes: [num of boxes, 4]
        # ind: [num of boxes]
        # pooled_features : [num of boxes, channels, pool_height, pool_width]
        pooled_features = RoIAlign_C.apply(feature_maps[i], level_boxes, ind, pool_size, pool_size, scale)

        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    # pooled: [total num of boxes, channels, pool_hegith, pool_width]
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    # pooled: [total num of boxes, channels, pool_hegith, pool_width]
    return pooled

class RoIHeader(nn.Module):
    def __init__(self, depth, pool_size, num_classes):
        super(RoIHeader, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois, image_shape):
        x = pyramid_roi_align(x, rois, self.pool_size, image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1,1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

class MaskHeader(nn.Module):
    def __init__(self, depth, pool_size, num_classes):
        super(MaskHeader, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, image_shape):
        x = pyramid_roi_align(x, rois, self.pool_size, image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x