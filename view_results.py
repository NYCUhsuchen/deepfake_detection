#!/usr/bin/env python3
"""
Quick script to view and analyze results
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix

def view_results(output_dir='./outputs'):
    """View and analyze experimental results"""
    output_path = Path(output_dir)
    
    print("🔍 Cross-Manipulation Deepfake Detection - Results Summary")
    print("=" * 70)
    
    # Check what files are available
    print("📁 Available files:")
    for file in sorted(output_path.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  📄 {file.name} ({size_mb:.2f} MB)")
    
    print()
    
    # Load and display metrics
    metrics_file = output_path / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print("📊 Performance Metrics:")
            print("-" * 30)
            for metric, value in metrics.items():
                print(f"  {metric:15s}: {value:.4f}")
            
            # Check if we meet the target
            auc = metrics.get('AUC', 0)
            if auc >= 0.80:
                print(f"\n✅ Target AUC achieved! ({auc:.3f} >= 0.80)")
            else:
                print(f"\n⚠️  AUC below target ({auc:.3f} < 0.80)")
                
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")
    else:
        print("❌ No metrics file found")
    
    print()
    
    # Load and analyze detailed results
    results_file = output_path / 'test_results.csv'
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            
            print("📈 Detailed Results Analysis:")
            print("-" * 40)
            print(f"  Total samples: {len(df)}")
            
            # Class distribution
            real_count = sum(df['true_label'] == 0)
            fake_count = sum(df['true_label'] == 1)
            print(f"  Real samples: {real_count}")
            print(f"  Fake samples: {fake_count}")
            
            # Performance breakdown
            correct = sum(df['true_label'] == df['predicted_label'])
            print(f"  Correctly classified: {correct} ({100*correct/len(df):.1f}%)")
            
            # Confusion matrix
            cm = confusion_matrix(df['true_label'], df['predicted_label'])
            print(f"\n  Confusion Matrix:")
            print(f"    Predicted:  Real  Fake")
            print(f"    Real:      {cm[0,0]:5d} {cm[0,1]:5d}")
            print(f"    Fake:      {cm[1,0]:5d} {cm[1,1]:5d}")
            
            # Error analysis
            false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
            false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
            
            print(f"\n  Error Analysis:")
            print(f"    False Positives (Real→Fake): {len(false_positives)}")
            print(f"    False Negatives (Fake→Real): {len(false_negatives)}")
            
            if len(false_positives) > 0:
                avg_fp_conf = false_positives['fake_probability'].mean()
                print(f"    Avg FP confidence: {avg_fp_conf:.3f}")
            
            if len(false_negatives) > 0:
                avg_fn_conf = false_negatives['fake_probability'].mean()
                print(f"    Avg FN confidence: {avg_fn_conf:.3f}")
            
            # Confidence distribution
            print(f"\n  Confidence Distribution:")
            print(f"    Mean fake probability: {df['fake_probability'].mean():.3f}")
            print(f"    Std fake probability: {df['fake_probability'].std():.3f}")
            print(f"    Min fake probability: {df['fake_probability'].min():.3f}")
            print(f"    Max fake probability: {df['fake_probability'].max():.3f}")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
    else:
        print("❌ No results file found")
    
    print()
    
    # Training log summary
    log_file = output_path / 'training.log'
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            print("📝 Training Summary:")
            print("-" * 25)
            
            # Find key information
            for line in lines:
                if "Total parameters:" in line:
                    print(f"  {line.strip()}")
                elif "Trainable parameters:" in line:
                    print(f"  {line.strip()}")
                elif "Training completed!" in line:
                    print(f"  ✅ Training completed successfully")
            
        except Exception as e:
            print(f"❌ Error reading training log: {e}")
    
    print()
    
    # Files for submission
    print("📦 Files for Assignment Submission:")
    print("-" * 40)
    
    required_files = [
        ('simplified_main.py', 'Main implementation'),
        ('requirements.txt', 'Dependencies'),
        ('README.md', 'Documentation'),
        ('outputs/final_model.pth', 'Trained model weights'),
        ('outputs/test_results.csv', 'Detailed results'),
        ('outputs/metrics.json', 'Performance metrics'),
        ('outputs/roc_curve.png', 'ROC curve visualization'),
        ('outputs/training.log', 'Training logs')
    ]
    
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path} - {description}")
        else:
            print(f"  ❌ {file_path} - {description} (MISSING)")

def create_summary_report(output_dir='./outputs'):
    """Create a summary report for the assignment"""
    output_path = Path(output_dir)
    
    # Load results
    metrics_file = output_path / 'metrics.json'
    results_file = output_path / 'test_results.csv'
    
    report_lines = []
    report_lines.append("Cross-Manipulation Deepfake Detection - Assignment Summary")
    report_lines.append("=" * 65)
    report_lines.append("")
    
    # Model architecture
    report_lines.append("Model Architecture:")
    report_lines.append("- Backbone: CLIP ViT-B/32 (frozen)")
    report_lines.append("- Adaptation: Feature adapter + text-visual fusion")
    report_lines.append("- Trainable parameters: <5% of total")
    report_lines.append("")
    
    # Results
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        report_lines.append("Performance Results:")
        for metric, value in metrics.items():
            report_lines.append(f"- {metric}: {value:.4f}")
        report_lines.append("")
    
    # Dataset
    if results_file.exists():
        df = pd.read_csv(results_file)
        report_lines.append("Dataset Information:")
        report_lines.append(f"- Training: Real_youtube + FaceSwap")
        report_lines.append(f"- Testing: {len(df)} samples (Real + NeuralTextures)")
        report_lines.append(f"- Cross-type evaluation: ✅ Achieved")
        report_lines.append("")
    
    # Key achievements
    report_lines.append("Key Achievements:")
    report_lines.append("✅ Parameter-efficient adaptation (<5% trainable)")
    report_lines.append("✅ Cross-type generalization (train≠test)")
    report_lines.append("✅ Text-visual semantic fusion")
    report_lines.append("✅ Reproducible implementation")
    
    if metrics_file.exists():
        auc = metrics.get('AUC', 0)
        if auc >= 0.80:
            report_lines.append("✅ Target AUC achieved (≥0.80)")
        else:
            report_lines.append("⚠️  AUC below target (<0.80)")
    
    # Save report
    with open(output_path / 'assignment_summary.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\n📄 Summary saved to {output_path / 'assignment_summary.txt'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View experimental results')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--create_summary', action='store_true', help='Create assignment summary')
    
    args = parser.parse_args()
    
    view_results(args.output_dir)
    
    if args.create_summary:
        create_summary_report(args.output_dir)
