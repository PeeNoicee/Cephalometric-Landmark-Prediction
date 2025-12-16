"""
HRNet (High-Resolution Network) for Cephalometric Landmark Detection
Based on: Deep High-Resolution Representation Learning for Visual Recognition
https://arxiv.org/abs/1908.07919

This is the architecture used by benchmark methods to achieve ~1.69mm MRE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for HRNet"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """Multi-resolution parallel convolution module"""
    def __init__(self, num_branches, block, num_blocks, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        layers = []
        layers.append(block(num_channels[branch_index], num_channels[branch_index], stride))
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)
    
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self, num_branches, num_channels):
        if num_branches == 1:
            return None
        
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_channels[i])
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        if self.fuse_layers is None:
            return x
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse


class HRNet(nn.Module):
    """
    HRNet for Cephalometric Landmark Detection
    
    Maintains high-resolution representations throughout the network,
    which is crucial for precise landmark localization.
    """
    def __init__(self, num_landmarks=29, num_cvm_stages=6, use_multitask=True):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_cvm_stages = num_cvm_stages
        self.use_multitask = use_multitask
        
        # Stem: Initial convolutions
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Transition 1: 256 -> [48, 96]
        self.transition1 = self._make_transition_layer([256], [48, 96])
        
        # Stage 2
        self.stage2 = self._make_stage(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=[48, 96]
        )
        
        # Transition 2: [48, 96] -> [48, 96, 192]
        self.transition2 = self._make_transition_layer([48, 96], [48, 96, 192])
        
        # Stage 3
        self.stage3 = self._make_stage(
            num_modules=4,
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=[48, 96, 192]
        )
        
        # Transition 3: [48, 96, 192] -> [48, 96, 192, 384]
        self.transition3 = self._make_transition_layer([48, 96, 192], [48, 96, 192, 384])
        
        # Stage 4
        self.stage4 = self._make_stage(
            num_modules=3,
            num_branches=4,
            num_blocks=[4, 4, 4, 4],
            num_channels=[48, 96, 192, 384]
        )
        
        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_landmarks, kernel_size=1)
        )
        
        # CVM classification head (if multitask)
        # Input: concatenated pooled features from all branches (720 channels)
        if use_multitask:
            self.cvm_head = nn.Sequential(
                nn.Linear(48 + 96 + 192 + 384, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_cvm_stages)
            )
        
        self._init_weights()
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        num_branches_pre = len(num_channels_pre)
        num_branches_cur = len(num_channels_cur)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre[-1] if j == 0 else num_channels_cur[i]
                    out_ch = num_channels_cur[i]
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            
            modules.append(
                HighResolutionModule(
                    num_branches,
                    BasicBlock,
                    num_blocks,
                    num_channels,
                    reset_multi_scale_output
                )
            )
        
        return nn.Sequential(*modules)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition 1
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        
        # Stage 2
        y_list = self.stage2(x_list)
        
        # Transition 2
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        
        # Stage 3
        y_list = self.stage3(x_list)
        
        # Transition 3
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        
        # Stage 4
        y_list = self.stage4(x_list)
        
        # Heatmap output (use highest resolution branch)
        heatmaps = self.heatmap_head(y_list[0])
        
        # CVM classification (if multitask)
        if self.use_multitask:
            # Aggregate features from all branches
            feat_list = []
            for feat in y_list:
                feat_list.append(F.adaptive_avg_pool2d(feat, 1))
            combined_feat = torch.cat(feat_list, dim=1)
            cvm_logits = self.cvm_head(combined_feat.view(combined_feat.size(0), -1))
            return heatmaps, cvm_logits
        
        return heatmaps


def build_hrnet(config):
    """Build HRNet model from config"""
    model = HRNet(
        num_landmarks=config.NUM_LANDMARKS,
        num_cvm_stages=config.NUM_CVM_STAGES,
        use_multitask=config.USE_MULTITASK
    )
    return model


if __name__ == '__main__':
    # Test the model
    model = HRNet(num_landmarks=29, num_cvm_stages=6, use_multitask=True)
    x = torch.randn(2, 1, 512, 512)
    heatmaps, cvm = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {heatmaps.shape}")
    print(f"CVM shape: {cvm.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
