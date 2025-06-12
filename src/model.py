import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as T
import torchvision
import timm

class VisualBackbone(nn.Module):
    def __init__(self, model_name, img_size=384, custom_weights=None):
        super().__init__()
        self.model_name = model_name
        self.custom_weights = custom_weights
        
        # Set default image size based on model
        if 'dinov2_vits14' in model_name:
            self.default_img_size = 840  # DINOv2 ViT-S/14 default for our experiments
        elif 'dinov2' in model_name:
            self.default_img_size = 518  # DINOv2 default
        elif 'dino' in model_name:
            self.default_img_size = 224  # DINO default
        else:
            self.default_img_size = 384  # Other models default
            
        # Adjust image size to be multiple of 14
        self.img_size = self._adjust_image_size(img_size)
        print(f"Using image size: {self.img_size} (adjusted to multiple of 14)")
        
        self._setup_model()
        
        # Flag to check if the model is a Vision Transformer (ViT)
        self.is_vit = False
        self.is_dinov2 = False  # Used for distinguishing between DINO and DINOv2
        
        # If the model is a ResNet, split it into different convolutional blocks
        if 'resnet' in model_name:
            self._split_resnet_layers()
        
    def _adjust_image_size(self, size):
        """Adjust image size to be multiple of 14."""
        if size % 14 != 0:
            # Round up to nearest multiple of 14
            adjusted_size = ((size + 13) // 14) * 14
            print(f"Warning: Image size {size} is not a multiple of 14. Adjusting to {adjusted_size}")
            return adjusted_size
        return size

    def _setup_model(self):
        if 'dinov2' in self.model_name:
            self._setup_dino_model()
        else:
            self._setup_other_transformer_models()

    def _setup_dino_model(self):
        if self.custom_weights:
            # Load custom DINO model with LoRA weights
            self.model = torch.load(self.custom_weights)
            if hasattr(self.model, 'module'):
                self.model = self.model.module
            print(f"Loaded custom DINO model from {self.custom_weights}")
            
            # Ensure the model is in eval mode
            self.model.eval()
            
            # For DINOv2 ViT-S/14, ensure patch size is 14
            if 'dinov2_vits14' in self.model_name:
                assert self.model.patch_embed.patch_size[0] == 14, \
                    f"Expected patch size 14 for DINOv2 ViT-S/14, got {self.model.patch_embed.patch_size[0]}"
        else:
            # Load pretrained DINOv2 model
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        
        self.model.eval()
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim
        
        # Set ViT and DINOv2 flags
        self.is_vit = True
        self.is_dinov2 = True

    def _setup_other_transformer_models(self):
        if 'mae' in self.model_name:
            self.model = timm.create_model(self.model_name, pretrained=True, img_size=self.img_size)
        elif 'clip' in self.model_name:
            self.model = timm.create_model(self.model_name, pretrained=True, img_size=self.img_size)
        elif 'sam' in self.model_name:
            self.model = timm.create_model(self.model_name, pretrained=True, img_size=self.img_size)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def _split_resnet_layers(self):
        """
        Split ResNet50 layers into different blocks.
        """
        children = list(self.model.children())
        self.conv1 = nn.Sequential(*children[:4])  # Initial convolution layers
        self.conv2 = children[4]  # Layer after the first conv block
        self.conv3 = children[5]  # After second conv block
        self.conv4 = children[6]  # After third conv block

    def _get_dino_patches_hook(self, module, input, output):
        """
        Hook function to store patch tokens for DINO model.
        """
        module.patch_outs = output
            
            
    def get_transform(self):
        """Get the appropriate transform for the model."""
        if 'dinov2_vits14' in self.model_name:
            # Specific transform for DINOv2 ViT-S/14
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Default transform for other models
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def forward(self, x):
        """
        Forward pass to extract features from input images.
        """
        feat = OrderedDict()

        # Handle DINOv2 model
        if 'dinov2' in self.model_name:
            # Get patch features for DINOv2
            patch_feats = self.model.forward_features(x)['x_norm_patchtokens']
            
            # Convert 1D patch tokens to 2D spatial grid
            n_patch_x_row = int(patch_feats.shape[1] ** 0.5)  # Assume square grid
            patch_feats = patch_feats.permute(0, 2, 1)  # Reorder dimensions
            patch_feats = patch_feats.view(patch_feats.shape[0], patch_feats.shape[1], n_patch_x_row, n_patch_x_row)  # Reshape to 2D grid
            feat['vit_out'] = patch_feats
            
        # Handle other transformer models (SAM, CLIP, MAE)
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'sam' in self.model_name:
            patch_feats = self.model.forward_features(x)[:, 1:, :]
            n_patch_x_row = int(patch_feats.shape[1] ** 0.5)
            patch_feats = patch_feats.permute(0, 2, 1)
            patch_feats = patch_feats.view(patch_feats.shape[0], patch_feats.shape[1], n_patch_x_row, n_patch_x_row)
            feat['vit_out'] = patch_feats
            
        # Handle ResNet models
        else:
            feat_map = self.conv1(x)
            feat_map = self.conv2(feat_map)
            feat_map3 = self.conv3(feat_map)
            feat_map4 = self.conv4(feat_map3)
            feat['map3'] = feat_map3
            feat['map4'] = feat_map4

        return feat
