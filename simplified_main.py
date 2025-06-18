#!/usr/bin/env python3
"""
Simplified Cross-Manipulation Deepfake Detection with CLIP
Safer version focusing on feature adaptation rather than LoRA
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import clip
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class CLIPDeepfakeDetector(nn.Module):
    def __init__(self, model_name="ViT-B/32", num_classes=2):
        super().__init__()
        
        # Load pretrained CLIP
        self.clip_model, self.preprocess = clip.load(model_name, device="cpu")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Add trainable adapter layers (parameter-efficient alternative to LoRA)
        self.feature_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 6, 256),  # 128 from adapter + 6 from text similarities
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Text prompts for deepfake detection
        self.text_prompts = [
            "a real human face",
            "a fake deepfake face",
            "an authentic person",
            "a manipulated face",
            "a genuine human",
            "a synthetic face"
        ]
        
        # Tokenize prompts
        self.text_tokens = clip.tokenize(self.text_prompts)
        
    def forward(self, images):
        batch_size = images.shape[0]
        device = images.device
        
        # Move text tokens to device
        text_tokens = self.text_tokens.to(device)
        
        # Extract image and text features using frozen CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity scores with text prompts
        similarity = image_features @ text_features.T  # [batch_size, num_prompts]
        
        # Aggregate similarities (real vs fake prompts)
        real_sim = similarity[:, [0, 2, 4]].mean(dim=1, keepdim=True)  # real prompts
        fake_sim = similarity[:, [1, 3, 5]].mean(dim=1, keepdim=True)  # fake prompts
        
        # Process image features through adapter
        adapted_features = self.feature_adapter(image_features)
        
        # Combine features
        combined_features = torch.cat([
            adapted_features,
            real_sim,
            fake_sim,
            similarity.mean(dim=1, keepdim=True),
            (real_sim - fake_sim),  # Real-fake difference
            similarity.std(dim=1, keepdim=True),  # Similarity variance
            similarity.max(dim=1, keepdim=True)[0]  # Max similarity
        ], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits, adapted_features

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir: Path, classes: List[str], transform=None, max_samples_per_class=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        logging.info(f"Creating dataset from {self.data_dir}")
        logging.info(f"Looking for classes: {classes}")
        
        for class_name in classes:
            class_dir = self.data_dir / class_name
            logging.info(f"Checking directory: {class_dir}")
            
            if not class_dir.exists():
                logging.error(f"Class directory {class_dir} does not exist!")
                continue
                
            # Get all image files (case-insensitive)
            image_files = []
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            
            for ext in extensions:
                found_files = list(class_dir.glob(ext))
                image_files.extend(found_files)
            
            if len(image_files) == 0:
                logging.warning(f"No image files found in {class_dir}")
                continue
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_path in image_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        logging.info(f"Dataset created with {len(self.samples)} samples")
        for class_name in classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            logging.info(f"  {class_name}: {count} samples")
        
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label, img_path

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate with robustness for edge cases"""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            logging.warning("Only one class present in y_true, cannot calculate EER properly")
            return 0.5, 0.5
        
        # Find the point where FPR and FNR are closest
        diff = np.absolute(fnr - fpr)
        
        # Handle all-NaN case
        if np.all(np.isnan(diff)):
            logging.warning("All differences are NaN, using default EER")
            return 0.5, 0.5
        
        min_idx = np.nanargmin(diff)
        eer_threshold = thresholds[min_idx] if min_idx < len(thresholds) else 0.5
        eer = fpr[min_idx]
        
        return eer, eer_threshold
    except Exception as e:
        logging.warning(f"Error calculating EER: {e}, using default values")
        return 0.5, 0.5

def evaluate_model(model, dataloader, device, save_results=False, results_path=None):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            all_paths.extend(paths)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    eer, eer_threshold = calculate_eer(all_labels, all_probs)
    
    # Convert probabilities to binary predictions using 0.5 threshold
    binary_preds = (np.array(all_probs) > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds)
    accuracy = accuracy_score(all_labels, binary_preds)
    
    metrics = {
        'AUC': auc,
        'EER': eer,
        'F1': f1,
        'Accuracy': accuracy,
        'EER_Threshold': eer_threshold
    }
    
    # Save results if requested
    if save_results and results_path:
        results_df = pd.DataFrame({
            'path': all_paths,
            'true_label': all_labels,
            'predicted_label': binary_preds,
            'fake_probability': all_probs
        })
        results_df.to_csv(results_path, index=False)
        logging.info(f"Results saved to {results_path}")
    
    return metrics, (all_labels, all_probs)

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve saved to {save_path}")
    
    plt.close()

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    """Train the model"""
    model.train()
    
    # Only optimize trainable parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Checkpoint saved: {checkpoint_path}')
        
        scheduler.step()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Cross-Manipulation Deepfake Detection (Simplified)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', help='Mode')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {args}")
    
    # Load model
    model = CLIPDeepfakeDetector(
        model_name="ViT-B/32",
        num_classes=2
    ).to(device)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    if args.mode in ['train', 'both']:
        # Training dataset (Real + FaceSwap)
        train_dataset = DeepfakeDataset(
            data_dir=args.data_dir,
            classes=['Real_youtube', 'FaceSwap'],
            transform=train_transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Train model
        logging.info("Starting training...")
        model = train_model(model, train_loader, None, device, args.epochs, args.lr)
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
        logging.info("Training completed!")
    
    if args.mode in ['test', 'both']:
        # Load best model if exists
        model_path = os.path.join(args.output_dir, 'final_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Loaded model from {model_path}")
        
        # Create a balanced test set: some Real samples + NeuralTextures
        # This gives us a proper binary classification evaluation
        
        # Real samples for testing (use a subset)
        real_test_dataset = DeepfakeDataset(
            data_dir=args.data_dir,
            classes=['Real_youtube'],
            transform=train_transform,
            max_samples_per_class=5000  # Use subset for testing
        )
        
        # NeuralTextures samples (fake)
        fake_test_dataset = DeepfakeDataset(
            data_dir=args.data_dir,
            classes=['NeuralTextures'],
            transform=train_transform
        )
        
        # Combine samples with correct labels
        test_samples = []
        
        # Add real samples (label 0)
        for path, _ in real_test_dataset.samples:
            test_samples.append((path, 0))
            
        # Add fake samples (label 1) 
        for path, _ in fake_test_dataset.samples:
            test_samples.append((path, 1))
        
        logging.info(f"Test set composition:")
        real_count = sum(1 for _, label in test_samples if label == 0)
        fake_count = sum(1 for _, label in test_samples if label == 1)
        logging.info(f"  Real samples: {real_count}")
        logging.info(f"  Fake samples (NeuralTextures): {fake_count}")
        logging.info(f"  Total test samples: {len(test_samples)}")
        
        # Create test dataset
        class TestDataset(Dataset):
            def __init__(self, samples, transform=None):
                self.samples = samples
                self.transform = transform
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, label, img_path
                except Exception as e:
                    logging.warning(f"Error loading image {img_path}: {e}")
                    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
                    if self.transform:
                        dummy_image = self.transform(dummy_image)
                    return dummy_image, label, img_path
        
        test_dataset = TestDataset(test_samples, train_transform)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        logging.info("Starting evaluation...")
        results_path = os.path.join(args.output_dir, 'test_results.csv')
        metrics, (y_true, y_scores) = evaluate_model(
            model, test_loader, device, save_results=True, results_path=results_path
        )
        
        # Log results
        logging.info("Test Results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Plot ROC curve
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(y_true, y_scores, roc_path)
        
        # Save metrics
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                metrics_json[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metrics_json[key] = int(value)
            else:
                metrics_json[key] = value
        
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=2)

if __name__ == "__main__":
    main()