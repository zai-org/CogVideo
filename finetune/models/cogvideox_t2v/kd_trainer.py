import torch
import torchvision
from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer
from ..utils import register
from typing_extensions import override


class CogVideoXT2VKdTrainer(CogVideoXT2VLoraTrainer):
    # Remove vae from the unload list to make it available in compute_loss
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = self.load_teacher_model()

    def load_teacher_model(self):
        # TODO: Replace with the actual path to the teacher model
        teacher_model_path = self.args.teacher_model_path if hasattr(self.args, 'teacher_model_path') else None
        if not teacher_model_path:
            print("Warning: teacher_model_path is not provided. Knowledge distillation will be skipped.")
            return None

        try:
            # Assuming the model is a torchvision Faster R-CNN model
            # The user should specify the number of classes in the model
            num_classes = self.args.teacher_model_num_classes if hasattr(self.args, 'teacher_model_num_classes') else 91 # COCO default
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
            # Load the pre-trained weights
            model.load_state_dict(torch.load(teacher_model_path))
            model.eval()
            model.to(self.accelerator.device)
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
        # The VAE is now available because we removed it from the UNLOAD_LIST
        video_frames = self.components.vae.decode(latents / self.components.vae.config.scaling_factor).sample

        # The output of the VAE is in the range [-1, 1]. We need to normalize it to [0, 1] for the teacher model.
        video_frames = (video_frames + 1) / 2

        # The video_frames tensor has shape [B, C, F, H, W]. We need to convert it to a list of frames for each video in the batch.
        # The shape should be [B, F, C, H, W]
        video_frames = video_frames.permute(0, 2, 1, 3, 4)


        # Calculate the knowledge distillation loss
        kd_loss = 0
        for i in range(video_frames.shape[0]): # For each video in the batch
            frames = [frame for frame in video_frames[i]] # list of frames for the i-th video
            teacher_output = self.teacher_model(frames)

            # The KD loss should encourage the presence of logos.
            # A simple loss could be based on the number of detected logos.
            # If no logos are detected, the loss is high.
            for output in teacher_output:
                if len(output['boxes']) == 0:
                    kd_loss += 1

        kd_loss /= (video_frames.shape[0] * video_frames.shape[1])

        # Combine the losses
        # The kd_loss_weight should be a hyperparameter defined in the args
        kd_loss_weight = self.args.kd_loss_weight if hasattr(self.args, 'kd_loss_weight') else 0.1
        total_loss = diffusion_loss + kd_loss_weight * kd_loss

        self.accelerator.log({"kd_loss": kd_loss, "diffusion_loss": diffusion_loss.item(), "total_loss": total_loss.item()})

        return total_loss

register("cogvideox-t2v", "kd", CogVideoXT2VKdTrainer)
