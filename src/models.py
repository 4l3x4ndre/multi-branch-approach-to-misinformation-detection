from collections import OrderedDict

import torch
from loguru import logger
import clip
from transformers import AutoImageProcessor, SiglipForImageClassification
from corpus_truth_manipulation.config import CONFIG
from src.demo_EGMMG import ClaimVerifier

class DeepFakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SiglipForImageClassification.from_pretrained(CONFIG.model.fakedetector_name).to(CONFIG.device)
        self.processor = AutoImageProcessor.from_pretrained(CONFIG.model.fakedetector_name)
        self.id2label = {"0": "fake", "1": "real"}

    def forward(self, images):
        images_processed = self.processor(images=images, return_tensors="pt").to(CONFIG.device)
        outputs = self.model.vision_model(**images_processed)            # [batch, num_patches, 768]
        image_features = torch.mean(outputs.last_hidden_state, dim=1)  # [batch, 768]

        return image_features


class CLIP(torch.nn.Module):
    def __init__(self, use_image_features=True, use_text_features=True):
        """
        Ablation options:
        - use_image_features: if False, ignore image features
        - use_text_features: if False, ignore text features
        """
        super().__init__()
        self.use_image_features = use_image_features
        self.use_text_features = use_text_features

        self.clip_model, self.preprocess = clip.load(CONFIG.model.clip_name, device=CONFIG.device)

    def forward(self, images, texts):
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(CONFIG.device)
        text_tensors = clip.tokenize(texts, truncate=True).to(CONFIG.device)

        image_features = self.clip_model.encode_image(img_tensors) if self.use_image_features else None
        text_features = self.clip_model.encode_text(text_tensors) if self.use_text_features else None

        # Combine features depending on which are enabled
        features = []
        if image_features is not None:
            features.append(image_features)
        if text_features is not None:
            features.append(text_features)

        return torch.hstack(features)


class MultimodalMisinformationDetector(torch.nn.Module):
    def __init__(
        self,
        use_deepfake_detector: bool = True,
        use_clip: bool = True,
        use_encyclopedic_knowledge: bool = True,
        classifier_input_dim: int = -1,
        clip_use_image_features:bool=True,
        clip_use_text_features:bool=True,
        ency_in_channels:int=773,
        ency_int_dims:int=1024,
        ency_hidden_dim:int=512,
        ency_conv:str="GAT",
        layer_sizes : list[int] = [],
        activation=torch.nn.ReLU,
        dropout: float = 0.0,
        activation_sigmoid:bool = True,
        finetune_deepfake_detector: bool = False,
    ):
        super().__init__()
        assert use_deepfake_detector or use_clip or use_encyclopedic_knowledge, "At least one feature extractor must be enabled."
        if use_clip:
            assert clip_use_image_features or clip_use_text_features, "At least one of clip_use_image_features or clip_use_text_features must be True if CLIP is enabled"

        # Ablation flags
        self.use_deepfake_detector = use_deepfake_detector
        self.use_clip = use_clip
        self.use_encyclopedic_knowledge = use_encyclopedic_knowledge
        self.finetune_deepfake_detector = finetune_deepfake_detector

        # Modules
        self.deepfake_detector = DeepFakeDetector() if use_deepfake_detector else None
        if self.use_deepfake_detector and not self.finetune_deepfake_detector:
            for param in self.deepfake_detector.parameters():
                param.requires_grad = False

        self.clip_model = CLIP(
            use_image_features=clip_use_image_features,
            use_text_features=clip_use_text_features
        ) if use_clip else None
        if self.use_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.encyclopedic_knowledge = ClaimVerifier(
            in_channels=ency_in_channels,
            int_dims=ency_int_dims,
            hidden_dim=ency_hidden_dim,
            conv=ency_conv,
            classifier=False
        ) if use_encyclopedic_knowledge else None

        # Classifier
        if len(layer_sizes) == 0:
            layer_sizes = [classifier_input_dim, 1024, 512, 256, 4]

        layers = OrderedDict()
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            layers[f'linear_{i}'] = torch.nn.Linear(in_size, out_size)

            # Add activation for all but the last layer
            if i < len(layer_sizes) - 2:
                layers[f'activation_{i}'] = activation()
                if dropout > 0:
                    layers[f'dropout_{i}'] = torch.nn.Dropout(dropout)
            elif activation_sigmoid:
                layers['final_activation'] = torch.nn.Sigmoid()

        self.classifier = torch.nn.Sequential(layers)

    def load_encycopedic_knowledge_weights(self, model_state_dict, **kwargs):
        if self.encyclopedic_knowledge is not None:
            self.encyclopedic_knowledge.load_state_dict(model_state_dict, **kwargs)
            logger.success("Loaded encyclopedic knowledge weights.")

    def set_mode(self,train:bool):
        if self.use_deepfake_detector:
            if self.finetune_deepfake_detector and train:
                self.deepfake_detector.train()
            else:
                self.deepfake_detector.eval()
        if self.use_clip:
            self.clip_model.eval()
        if self.use_encyclopedic_knowledge:
            if train:
                self.encyclopedic_knowledge.train()
            else:
                self.encyclopedic_knowledge.eval()

    def forward(self, image, text_str, claim_data, evidence_data):
        features_list = []

        if self.use_deepfake_detector:
            features_list.append(self.deepfake_detector(image))
        if self.use_clip:
            features_list.append(self.clip_model(image, text_str))

        if self.use_encyclopedic_knowledge:
            features_list.append(
                self.encyclopedic_knowledge(claim_data=claim_data, evidence_data=evidence_data)
            )

        # If no features are enabled, return zeros
        if not features_list:
            raise ValueError("At least one feature extractor must be enabled.")
        else:
            combined_features = torch.cat(features_list, dim=1).float()

        return self.classifier(combined_features)



