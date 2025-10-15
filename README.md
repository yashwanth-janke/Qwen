# Qwen3 LoRA Fine-Tuning Pipeline

A production-ready implementation for fine-tuning Qwen3 language models on AWS SageMaker using Parameter-Efficient Fine-Tuning techniques.

---

## Project Overview

This repository contains an end-to-end machine learning pipeline for fine-tuning Qwen3 models with LoRA (Low-Rank Adaptation) and deploying them on Amazon SageMaker. The project emphasizes memory efficiency, scalability, and Chain-of-Thought reasoning enhancement.

**Key Features:**
- Memory-optimized training with 4-bit quantization
- Parameter-efficient fine-tuning using LoRA adapters
- Distributed training support for multi-GPU setups
- Side-by-side model comparison (base vs fine-tuned)
- Production-ready deployment with SageMaker Inference Components

---

## Project Structure

```
Qwen/

 1-1. Preparation.ipynb         # Step 1: Environment & data setup
 1-2. Fine-tuning-model.ipynb   # Step 2: Model training with LoRA
 1-3. Serving-model.ipynb       # Step 3: Deployment & evaluation
 test.py                         # Testing utilities
 README.md                       # You are here

 src/
     sm_lora_trainer.py         # Core training script
     requirements.txt            # Python dependencies
     configs/
         qwen3-4b.yaml          # Hyperparameter configuration
```

---

## Workflow Guide

### Notebook 1: Preparation
Set up your environment and prepare everything needed for training.

- Install ML dependencies (transformers, PEFT, TRL, bitsandbytes)
- Configure Docker settings for optimal performance
- Download Qwen3-4B model from Hugging Face
- Prepare Chain-of-Thought reasoning dataset
- Upload assets to S3 buckets

### Notebook 2: Fine-Tuning
Train your model using memory-efficient LoRA techniques.

- Configure LoRA hyperparameters (rank, alpha, target modules)
- Launch distributed training jobs on SageMaker
- Monitor training progress and metrics
- Merge LoRA adapters with base model
- Package optimized model artifacts

### Notebook 3: Serving
Deploy and evaluate your trained model.

- Create SageMaker endpoints with auto-scaling
- Deploy multiple model variants simultaneously
- Run Chain-of-Thought reasoning tests
- Compare base vs fine-tuned performance
- Clean up resources to avoid unnecessary costs

---

## Setup and Requirements

### AWS Prerequisites
- Active AWS account with SageMaker enabled
- IAM role with permissions for SageMaker, S3, and ECR
- SageMaker Notebook Instance (recommended: ml.t3.medium or higher)

### Compute Resources
| Purpose | Recommended Instance | GPU |
|---------|---------------------|-----|
| Training | ml.g5.2xlarge | 1x NVIDIA A10 (24GB) |
| Inference | ml.g5.2xlarge - ml.g5.4xlarge | 1-2x NVIDIA A10 |
| Development | ml.t3.medium | CPU only |

### Software Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- AWS SageMaker SDK 2.150.0+

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 4.51.3 | Hugging Face model infrastructure |
| peft | 0.15.2 | LoRA implementation and PEFT methods |
| trl | 0.17.0 | Training utilities for language models |
| bitsandbytes | 0.45.5 | 4-bit quantization for memory efficiency |
| accelerate | 1.2.1 | Multi-GPU distributed training |
| datasets | 3.5.1 | Dataset loading and preprocessing |
| sagemaker | >=2.150.0 | AWS SageMaker integration |
| torch | >=2.0.0 | Deep learning framework |

Full dependency list available in `src/requirements.txt`.

---

## Configuration

### Training Hyperparameters
Configuration file: `src/configs/qwen3-4b.yaml`

```yaml
learning_rate: 2e-3           # Optimized for LoRA fine-tuning
lora_rank: 64                 # Rank of adaptation matrices
lora_alpha: 128               # Scaling factor for LoRA
batch_size: 4                 # Per-device batch size
gradient_accumulation: 4      # Effective batch size multiplier
max_seq_length: 2048          # Maximum sequence length
```

### LoRA Target Modules
The implementation targets key attention and MLP layers:
- Query/Key/Value projections (q_proj, k_proj, v_proj)
- Output projections (o_proj)
- Feed-forward networks (gate_proj, up_proj, down_proj)

---

## Features and Optimizations

### Memory Efficiency
- **4-bit Quantization**: Reduces model memory footprint by approximately 75%
- **Gradient Checkpointing**: Trades compute for memory during backpropagation
- **Dynamic Padding**: Minimizes wasted computation on short sequences
- **Disk Offloading**: Handles models larger than GPU memory

### Training Capabilities
- **Distributed Training**: Multi-GPU support via PyTorch distributed backend
- **Mixed Precision**: Automatic mixed precision for faster training
- **Checkpoint Management**: Automatic saving of best models
- **Logging**: Comprehensive metrics tracking

### Deployment Features
- **Inference Components**: Host multiple models on single endpoint
- **Auto-scaling**: Dynamic resource allocation based on traffic
- **A/B Testing**: Compare model variants in production
- **Cost Optimization**: Efficient resource utilization

---

## Quick Start

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
   1-1. Preparation.ipynb  Setup environment
   1-2. Fine-tuning-model.ipynb  Train model
   1-3. Serving-model.ipynb  Deploy & evaluate
   ```

4. **Monitor and evaluate**
   - Check training metrics in CloudWatch
   - Test inference endpoints
   - Compare model performance

---

## Evaluation Metrics

The pipeline tracks multiple performance indicators:

- **Training Metrics**: Loss, learning rate, gradient norms
- **Inference Latency**: Response time for different model sizes
- **Quality Assessment**: Chain-of-Thought reasoning accuracy
- **Resource Usage**: Memory consumption, GPU utilization
- **Cost Analysis**: Training and inference expenses

---

## Troubleshooting

### Common Issues

**Out of Memory errors during training**
- Reduce per_device_train_batch_size in config
- Increase gradient_accumulation_steps
- Enable gradient checkpointing
- Use smaller LoRA rank

**Slow training speed**
- Check GPU utilization with nvidia-smi
- Ensure data is loaded from S3 correctly
- Verify distributed training setup
- Monitor CloudWatch logs for bottlenecks

**Inference endpoint timeout**
- Increase endpoint timeout settings
- Use larger instance types
- Enable auto-scaling for high traffic

---

## Performance Tips

1. **Optimize Batch Size**: Find the sweet spot between memory usage and throughput
2. **Use Mixed Precision**: Enable fp16/bf16 for faster training
3. **Data Loading**: Pre-process and cache datasets to avoid I/O bottlenecks
4. **LoRA Configuration**: Experiment with rank and alpha values for your use case
5. **Instance Selection**: Use spot instances for cost-effective training

---

## Useful Resources

- [Qwen3 Model Documentation](https://huggingface.co/Qwen)
- [LoRA Paper: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Transformers Library](https://huggingface.co/docs/transformers/index)

---

## License

This project is provided as-is for educational and research purposes.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## Author

**Yashwanth Janke**
- GitHub: [@yashwanth-janke](https://github.com/yashwanth-janke)

---

## Acknowledgments

Special thanks to:
- The open-source ML community
- Hugging Face team for transformers and PEFT libraries
- AWS for SageMaker platform
- PyTorch development team

---

**Built with care for the ML community**
