# Qwen3 Fine-Tuning on Amazon SageMaker

This project provides a complete end-to-end workflow for fine-tuning Qwen3 models using LoRA (Low-Rank Adaptation) techniques and deploying them for inference on Amazon SageMaker. The implementation focuses on Chain-of-Thought reasoning capabilities with memory-optimized training.

## Overview

The workflow demonstrates how to:
- Set up the environment and prepare Qwen3 models for fine-tuning
- Fine-tune models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Deploy and compare pre-trained vs fine-tuned models using SageMaker Inference Components
- Evaluate Chain-of-Thought reasoning performance across different model variants

## Directory Structure

```
├── 1-1. Preparation.ipynb              # Environment setup and data preparation
├── 1-2. Fine-tuning-model.ipynb        # LoRA-based model fine-tuning
├── 1-3. Serving-model.ipynb            # Model deployment and inference comparison
└── src/
    ├── sm_lora_trainer.py              # Main training script with distributed support
    ├── configs/
    │   └── qwen3-4b.yaml               # Training configuration file
    └── requirements.txt                # Python dependencies
```

## Notebook Descriptions

### 1-1. Preparation.ipynb
- **Environment Setup**: Install required ML libraries and configure Docker for optimal performance
- **Model Download**: Download Qwen3-4B model and tokenizer from Hugging Face Hub
- **Data Preparation**: Format Chain-of-Thought reasoning dataset with custom prompt templates
- **S3 Upload**: Prepare and upload training data and model artifacts to Amazon S3

### 1-2. Fine-tuning-model.ipynb
- **Training Configuration**: Set up memory-optimized hyperparameters for efficient fine-tuning
- **LoRA Implementation**: Configure Parameter-Efficient Fine-Tuning with target module selection
- **SageMaker Training**: Execute distributed training jobs with PyTorch framework
- **Model Merging**: Combine base model with LoRA weights for inference deployment
- **Compression**: Prepare optimized model artifacts for production deployment

### 1-3. Serving-model.ipynb
- **Endpoint Setup**: Create SageMaker endpoints with auto-scaling configuration
- **Inference Components**: Deploy both pre-trained and fine-tuned models simultaneously
- **Performance Testing**: Run comprehensive Chain-of-Thought reasoning evaluations
- **Comparison Analysis**: Side-by-side comparison of model performance and response quality
- **Resource Cleanup**: Systematic cleanup of deployed resources to manage costs

## Getting Started

### Prerequisites
- AWS Account with SageMaker access
- SageMaker Notebook Instance (ml.t3.medium or larger recommended)
- IAM role with SageMaker, S3, and ECR permissions

### Installation
1. Clone this repository to your SageMaker Notebook Instance
2. Run the notebooks in sequence (1-1 → 1-2 → 1-3)
3. The first notebook will automatically install all required dependencies

### Runtime Environment
- **Python**: 3.10 or higher
- **Recommended Training Instance**: `ml.g5.2xlarge` (1x A10 GPU)
- **Recommended Inference Instance**: `ml.g5.2xlarge` or `ml.g5.4xlarge`
- **Framework**: PyTorch 2.3.0 with CUDA support

## Key Features

### Memory Optimization
- 4-bit quantization with BitsAndBytes
- Gradient checkpointing for reduced memory usage
- Dynamic padding and efficient data loading
- Disk offloading for large model handling

### Training Configuration
- LoRA rank and alpha parameter tuning
- Target module selection for optimal adaptation
- Distributed training support with torchrun
- Comprehensive logging and checkpoint management

### Chain-of-Thought Enhancement
- Custom prompt templates for structured reasoning
- Step-by-step thinking process before final answers
- Evaluation across diverse question types and domains
- Performance comparison between model variants

## Core Dependencies

```
transformers==4.51.3       # Hugging Face model library
datasets==3.5.1           # Dataset processing and management
peft==0.15.2              # Parameter-Efficient Fine-Tuning
trl==0.17.0               # Transformer Reinforcement Learning
bitsandbytes==0.45.5      # Quantization and memory optimization
accelerate==1.2.1         # Distributed training acceleration
sagemaker>=2.150.0        # AWS SageMaker SDK
torch>=2.0.0              # PyTorch deep learning framework
```

## Configuration Options

### Training Parameters
- **Batch Size**: Configurable per-device batch size with gradient accumulation
- **Learning Rate**: Optimized for LoRA fine-tuning (default: 2e-3)
- **LoRA Rank**: Adjustable adaptation rank (default: 64)
- **Target Modules**: Customizable module selection for adaptation

### Instance Types
- **Training**: ml.g5.2xlarge, ml.g5.4xlarge, ml.p4d.24xlarge
- **Inference**: ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.12xlarge
- **Development**: Local GPU support for testing

## Monitoring and Evaluation

The project includes comprehensive evaluation metrics:
- Response time measurement across different model variants
- Chain-of-Thought reasoning quality assessment
- Memory usage optimization tracking
- Cost analysis and resource utilization monitoring

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/index)
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Original Author: YJ Choi](https://github.com/Napkin-DL/qwen3-on-sagemaker/tree/main)