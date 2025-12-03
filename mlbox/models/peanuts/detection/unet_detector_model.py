from typing import List
import torch
import cv2
import numpy as np
import supervision as sv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from mlbox.models.peanuts.detection.abstract_detection_model import AbstractPeanutsDetector


class UNetPeanutsDetector(AbstractPeanutsDetector):
    def __init__(self, weights_path: str, device: str = None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model (same architecture as training)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # Don't load pretrained, we'll load our weights
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform (same as training validation)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def detect(self, images: List[np.ndarray], *args, **kwargs) -> List[sv.Detections]:
        """
        Detect peanuts in images using UNet segmentation.
        
        Args:
            images: List of numpy arrays (RGB images)
            *args, **kwargs: Ignored (for compatibility with YOLO interface)
            
        Returns:
            List of sv.Detections objects, each containing a mask
        """
        sv_detections = []
        
        for image in images:
            # Store original size
            h_orig, w_orig = image.shape[:2]
            
            # Images coming from peanuts service are already RGB (PIL -> np.array),
            # same as in the training and assessment scripts, so we pass them directly.
            transformed = self.transform(image=image)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img_tensor)
                pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Resize mask back to original image size
            pred_mask_resized = cv2.resize(
                pred_mask, 
                (w_orig, h_orig), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert to boolean mask
            pred_mask_bool = pred_mask_resized.astype(bool)
            
            # Create sv.Detections object
            # UNet outputs a single mask per image, so we wrap it in the expected format
            detection = sv.Detections(
                xyxy=np.array([[0, 0, w_orig, h_orig]]),  # Full image bbox
                mask=np.array([pred_mask_bool]),  # List of masks
                confidence=np.array([1.0]),  # Dummy confidence
            )
            
            sv_detections.append(detection)
        
        return sv_detections

