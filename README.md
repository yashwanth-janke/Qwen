# Qwen3 LoRA Fine-Tuning Pipeline# Qwen3 Fine-Tuning on Amazon SageMaker



**A production-ready implementation for fine-tuning Qwen3 language models on AWS SageMaker using Parameter-Efficient Fine-Tuning techniques.**This project provides a complete end-to-end workflow for fine-tuning Qwen3 models using LoRA (Low-Rank Adaptation) techniques and deploying them for inference on Amazon SageMaker. The implementation focuses on Chain-of-Thought reasoning capabilities with memory-optimized training.



---## Overview



## 🚀 Project OverviewThe workflow demonstrates how to:

- Set up the environment and prepare Qwen3 models for fine-tuning

This repository contains an end-to-end machine learning pipeline for fine-tuning Qwen3 models with LoRA (Low-Rank Adaptation) and deploying them on Amazon SageMaker. The project emphasizes memory efficiency, scalability, and Chain-of-Thought reasoning enhancement.- Fine-tune models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA

- Deploy and compare pre-trained vs fine-tuned models using SageMaker Inference Components

**What makes this implementation special:**- Evaluate Chain-of-Thought reasoning performance across different model variants

- ✅ Memory-optimized training with 4-bit quantization

- ✅ Parameter-efficient fine-tuning using LoRA adapters## Directory Structure

- ✅ Distributed training support for multi-GPU setups

- ✅ Side-by-side model comparison (base vs fine-tuned)```

- ✅ Production-ready deployment with SageMaker Inference Components├── 1-1. Preparation.ipynb              # Environment setup and data preparation

├── 1-2. Fine-tuning-model.ipynb        # LoRA-based model fine-tuning

---├── 1-3. Serving-model.ipynb            # Model deployment and inference comparison

└── src/

## 📁 Project Structure    ├── sm_lora_trainer.py              # Main training script with distributed support

    ├── configs/

```    │   └── qwen3-4b.yaml               # Training configuration file

Qwen/    └── requirements.txt                # Python dependencies

│```

├── 1-1. Preparation.ipynb         # Step 1: Environment & data setup

├── 1-2. Fine-tuning-model.ipynb   # Step 2: Model training with LoRA## Notebook Descriptions

├── 1-3. Serving-model.ipynb       # Step 3: Deployment & evaluation

├── test.py                         # Testing utilities### 1-1. Preparation.ipynb

├── README.md                       # You are here- **Environment Setup**: Install required ML libraries and configure Docker for optimal performance

│- **Model Download**: Download Qwen3-4B model and tokenizer from Hugging Face Hub

└── src/- **Data Preparation**: Format Chain-of-Thought reasoning dataset with custom prompt templates

    ├── sm_lora_trainer.py         # Core training script- **S3 Upload**: Prepare and upload training data and model artifacts to Amazon S3

    ├── requirements.txt            # Python dependencies

    └── configs/### 1-2. Fine-tuning-model.ipynb

        └── qwen3-4b.yaml          # Hyperparameter configuration- **Training Configuration**: Set up memory-optimized hyperparameters for efficient fine-tuning

```- **LoRA Implementation**: Configure Parameter-Efficient Fine-Tuning with target module selection

- **SageMaker Training**: Execute distributed training jobs with PyTorch framework

---- **Model Merging**: Combine base model with LoRA weights for inference deployment

- **Compression**: Prepare optimized model artifacts for production deployment

## 📚 Workflow Guide

### 1-3. Serving-model.ipynb

### **Notebook 1: Preparation**- **Endpoint Setup**: Create SageMaker endpoints with auto-scaling configuration

Set up your environment and prepare everything needed for training.- **Inference Components**: Deploy both pre-trained and fine-tuned models simultaneously

- **Performance Testing**: Run comprehensive Chain-of-Thought reasoning evaluations

- Install ML dependencies (transformers, PEFT, TRL, bitsandbytes)- **Comparison Analysis**: Side-by-side comparison of model performance and response quality

- Configure Docker settings for optimal performance- **Resource Cleanup**: Systematic cleanup of deployed resources to manage costs

- Download Qwen3-4B model from Hugging Face

- Prepare Chain-of-Thought reasoning dataset## Getting Started

- Upload assets to S3 buckets

### Prerequisites

### **Notebook 2: Fine-Tuning**- AWS Account with SageMaker access

Train your model using memory-efficient LoRA techniques.- SageMaker Notebook Instance (ml.t3.medium or larger recommended)

- IAM role with SageMaker, S3, and ECR permissions

- Configure LoRA hyperparameters (rank, alpha, target modules)

- Launch distributed training jobs on SageMaker### Installation

- Monitor training progress and metrics1. Clone this repository to your SageMaker Notebook Instance

- Merge LoRA adapters with base model2. Run the notebooks in sequence (1-1 → 1-2 → 1-3)

- Package optimized model artifacts3. The first notebook will automatically install all required dependencies



### **Notebook 3: Serving**### Runtime Environment

Deploy and evaluate your trained model.- **Python**: 3.10 or higher

- **Recommended Training Instance**: `ml.g5.2xlarge` (1x A10 GPU)

- Create SageMaker endpoints with auto-scaling- **Recommended Inference Instance**: `ml.g5.2xlarge` or `ml.g5.4xlarge`

- Deploy multiple model variants simultaneously- **Framework**: PyTorch 2.3.0 with CUDA support

- Run Chain-of-Thought reasoning tests

- Compare base vs fine-tuned performance## Key Features

- Clean up resources to avoid unnecessary costs

### Memory Optimization

---- 4-bit quantization with BitsAndBytes

- Gradient checkpointing for reduced memory usage

## ⚙️ Setup & Requirements- Dynamic padding and efficient data loading

- Disk offloading for large model handling

### **AWS Prerequisites**

- Active AWS account with SageMaker enabled### Training Configuration

- IAM role with permissions for SageMaker, S3, and ECR- LoRA rank and alpha parameter tuning

- SageMaker Notebook Instance (recommended: `ml.t3.medium` or higher)- Target module selection for optimal adaptation

- Distributed training support with torchrun

### **Compute Resources**- Comprehensive logging and checkpoint management

| Purpose | Recommended Instance | GPU |

|---------|---------------------|-----|### Chain-of-Thought Enhancement

| Training | `ml.g5.2xlarge` | 1x NVIDIA A10 (24GB) |- Custom prompt templates for structured reasoning

| Inference | `ml.g5.2xlarge` - `ml.g5.4xlarge` | 1-2x NVIDIA A10 |- Step-by-step thinking process before final answers

| Development | `ml.t3.medium` | CPU only |- Evaluation across diverse question types and domains

- Performance comparison between model variants

### **Software Requirements**

- Python 3.10+## Core Dependencies

- PyTorch 2.0+ with CUDA support

- AWS SageMaker SDK 2.150.0+```

transformers==4.51.3       # Hugging Face model library

---datasets==3.5.1           # Dataset processing and management

peft==0.15.2              # Parameter-Efficient Fine-Tuning

## 📦 Key Dependenciestrl==0.17.0               # Transformer Reinforcement Learning

bitsandbytes==0.45.5      # Quantization and memory optimization

| Package | Version | Purpose |accelerate==1.2.1         # Distributed training acceleration

|---------|---------|---------|sagemaker>=2.150.0        # AWS SageMaker SDK

| `transformers` | 4.51.3 | Hugging Face model infrastructure |torch>=2.0.0              # PyTorch deep learning framework

| `peft` | 0.15.2 | LoRA implementation & PEFT methods |```

| `trl` | 0.17.0 | Training utilities for language models |

| `bitsandbytes` | 0.45.5 | 4-bit quantization for memory efficiency |## Configuration Options

| `accelerate` | 1.2.1 | Multi-GPU distributed training |

| `datasets` | 3.5.1 | Dataset loading and preprocessing |### Training Parameters

| `sagemaker` | >=2.150.0 | AWS SageMaker integration |- **Batch Size**: Configurable per-device batch size with gradient accumulation

| `torch` | >=2.0.0 | Deep learning framework |- **Learning Rate**: Optimized for LoRA fine-tuning (default: 2e-3)

- **LoRA Rank**: Adjustable adaptation rank (default: 64)

Full dependency list available in `src/requirements.txt`.- **Target Modules**: Customizable module selection for adaptation



---### Instance Types

- **Training**: ml.g5.2xlarge, ml.g5.4xlarge, ml.p4d.24xlarge

## 🔧 Configuration- **Inference**: ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.12xlarge

- **Development**: Local GPU support for testing

### **Training Hyperparameters** (`src/configs/qwen3-4b.yaml`)

```yaml## Monitoring and Evaluation

learning_rate: 2e-3           # Optimized for LoRA fine-tuning

lora_rank: 64                 # Rank of adaptation matricesThe project includes comprehensive evaluation metrics:

lora_alpha: 128               # Scaling factor for LoRA- Response time measurement across different model variants

batch_size: 4                 # Per-device batch size- Chain-of-Thought reasoning quality assessment

gradient_accumulation: 4      # Effective batch size multiplier- Memory usage optimization tracking

max_seq_length: 2048          # Maximum sequence length- Cost analysis and resource utilization monitoring

```

### **LoRA Target Modules**
The implementation targets key attention and MLP layers:
- Query/Key/Value projections (`q_proj`, `k_proj`, `v_proj`)
- Output projections (`o_proj`)
- Feed-forward networks (`gate_proj`, `up_proj`, `down_proj`)

---

## 🎯 Features & Optimizations

### **Memory Efficiency**
- **4-bit Quantization**: Reduces model memory footprint by ~75%
- **Gradient Checkpointing**: Trades compute for memory during backpropagation
- **Dynamic Padding**: Minimizes wasted computation on short sequences
- **Disk Offloading**: Handles models larger than GPU memory

### **Training Capabilities**
- **Distributed Training**: Multi-GPU support via PyTorch's distributed backend
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
- **Checkpoint Management**: Automatic saving of best models
- **Logging**: Comprehensive metrics tracking with Weights & Biases integration

### **Deployment Features**
- **Inference Components**: Host multiple models on single endpoint
- **Auto-scaling**: Dynamic resource allocation based on traffic
- **A/B Testing**: Compare model variants in production
- **Cost Optimization**: Efficient resource utilization

---

## 🚦 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashwanth-janke/Qwen.git
   cd Qwen
   ```

2. **Launch SageMaker Notebook Instance**
   - Navigate to AWS SageMaker Console
   - Create a notebook instance with appropriate IAM role
   - Open JupyterLab

3. **Run notebooks in sequence**
   ```
   1-1. Preparation.ipynb → Setup environment
   1-2. Fine-tuning-model.ipynb → Train model
   1-3. Serving-model.ipynb → Deploy & evaluate
   ```

4. **Monitor and evaluate**
   - Check training metrics in CloudWatch
   - Test inference endpoints
   - Compare model performance

---

## 📊 Evaluation Metrics

The pipeline tracks multiple performance indicators:

- **Training Metrics**: Loss, learning rate, gradient norms
- **Inference Latency**: Response time for different model sizes
- **Quality Assessment**: Chain-of-Thought reasoning accuracy
- **Resource Usage**: Memory consumption, GPU utilization
- **Cost Analysis**: Training and inference expenses

---

## 🔗 Useful Resources

- [Qwen3 Model Documentation](https://huggingface.co/Qwen)
- [LoRA Paper: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Transformers Library](https://huggingface.co/docs/transformers/index)

---

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory (OOM) errors during training**
- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing
- Use smaller LoRA rank

**Slow training speed**
- Check GPU utilization with `nvidia-smi`
- Ensure data is loaded from S3 correctly
- Verify distributed training setup
- Monitor CloudWatch logs for bottlenecks

**Inference endpoint timeout**
- Increase endpoint timeout settings
- Use larger instance types
- Enable auto-scaling for high traffic

---

## 📈 Performance Tips

1. **Optimize Batch Size**: Find the sweet spot between memory usage and throughput
2. **Use Mixed Precision**: Enable fp16/bf16 for faster training
3. **Data Loading**: Pre-process and cache datasets to avoid I/O bottlenecks
4. **LoRA Configuration**: Experiment with rank and alpha values for your use case
5. **Instance Selection**: Use spot instances for cost-effective training

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 👨‍💻 Author

**Yashwanth Janke**
- GitHub: [@yashwanth-janke](https://github.com/yashwanth-janke)

---

## 🙏 Acknowledgments

Special thanks to:
- The open-source ML community
- Hugging Face team for transformers and PEFT libraries
- AWS for SageMaker platform
- PyTorch development team

---

**Built with ❤️ for the ML community**
