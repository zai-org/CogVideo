import torch
import torchvision
from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer
from ..utils import register
from typing_extensions import override
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
import torch.nn as nn
from collections import OrderedDict

class VGG16BackboneWrapper(nn.Module):
    def __init__(self, vgg_features):
        super(VGG16BackboneWrapper, self).__init__()
        self.features = vgg_features
        self.out_channels = 512

    def forward(self, x):
        x = self.features(x)
        return OrderedDict([("0", x)])

class CogVideoXT2VKdTrainer(CogVideoXT2VLoraTrainer):
    # Remove vae from the unload list to make it available in compute_loss
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = self.load_teacher_model()

    def load_teacher_model(self):
        teacher_model_path = self.args.teacher_model_path if hasattr(self.args, 'teacher_model_path') else None
        if not teacher_model_path:
            print("Warning: teacher_model_path is not provided. Knowledge distillation will be skipped.")
            return None

        try:
            # Create a VGG16-based Faster R-CNN model
            # 1. VGG16 backbone
            vgg_features = vgg16(weights=None).features
            # The original VGG16 model in torchvision has a maxpool layer at the end of features.
            # Faster R-CNN with VGG backbone in many implementations does not use this last maxpool.
            # Let's remove it to be closer to the original Caffe implementation.
            backbone_features = vgg_features[:-1]
            backbone = VGG16BackboneWrapper(backbone_features)

            # 2. RPN
            anchor_generator = AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

            # 3. RoI heads
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

            # The user's model has fc6 and fc7 layers, which corresponds to a TwoMLPHead.
            # VGG16's output from the backbone is 512 * 7 * 7 = 25088
            box_head = TwoMLPHead(in_channels=25088, representation_size=4096)

            num_classes = self.args.teacher_model_num_classes if hasattr(self.args, 'teacher_model_num_classes') else 91 # COCO default
            box_predictor = FastRCNNPredictor(in_channels=4096, num_classes=num_classes)


            # 4. Faster R-CNN model
            model = FasterRCNN(backbone,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler,
                               box_head=box_head,
                               box_predictor=box_predictor,
                               num_classes=num_classes)

            # Load the pre-trained weights from the converted file
            print(f"Loading teacher model from: {teacher_model_path}")
            state_dict = torch.load(teacher_model_path)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.accelerator.device)
            print("Teacher model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading teacher model: {e}")
            return None

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # Get the original diffusion loss
        diffusion_loss = super().compute_loss(batch)

        if self.teacher_model is None:
            return diffusion_loss

        latents = batch["encoded_videos"]

        # Decode the latents to get video frames
        video_frames = self.components.vae.decode(latents / self.components.vae.config.scaling_factor).sample

        # The output of the VAE is in the range [-1, 1]. We need to normalize it to [0, 1] for the teacher model.
        video_frames = (video_frames + 1) / 2

        video_frames = video_frames.permute(0, 2, 1, 3, 4)

        # Calculate the knowledge distillation loss
        kd_loss = 0
        num_frames_processed = 0
        for i in range(video_frames.shape[0]): # For each video in the batch
            frames = [frame for frame in video_frames[i]] # list of frames for the i-th video
            if not frames:
                continue

            num_frames_processed += len(frames)
            teacher_output = self.teacher_model(frames)

            for output in teacher_output:
                if len(output['boxes']) == 0:
                    kd_loss += 1

        if num_frames_processed > 0:
            kd_loss /= num_frames_processed
        else:
            kd_loss = 0

        # Combine the losses
        kd_loss_weight = self.args.kd_loss_weight if hasattr(self.args, 'kd_loss_weight') else 0.1
        # Make kd_loss a tensor
        kd_loss_tensor = torch.tensor(kd_loss, device=self.accelerator.device, dtype=diffusion_loss.dtype)
        total_loss = diffusion_loss + kd_loss_weight * kd_loss_tensor

        self.accelerator.log({"kd_loss": kd_loss, "diffusion_loss": diffusion_loss.item(), "total_loss": total_loss.item()})

        return total_loss

register("cogvideox-t2v", "kd", CogVideoXT2VKdTrainer)
