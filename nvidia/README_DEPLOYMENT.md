# RTX 5090 Parameter Sweep Deployment Guide

## 🚀 Quick Deployment

Deploy the parameter sweep to your RTX 5090 server:

```bash
# From the project root directory
./deploy-to-galactica.sh nvidia diffy-gpu-sweep
```

## 📁 What Gets Deployed

The deployment script will:

1. **Copy** the entire `nvidia/` folder to the RTX 5090 server
2. **Build** the Docker image with RTX 5090 support (CUDA 12.4)
3. **Run** RTX 5090 compatibility tests
4. **Execute** 20-hour parameter sweep with hundreds of experiments
5. **Download** all results back to your local machine

## 📊 Volume Mapping

Your deployment script automatically mounts these volumes:

```bash
# Models and checkpoints
/app/models -> ~/models/

# Training logs and detailed results
/app/logs -> ~/logs/

# Summary results for download
/app/generated_images -> ~/simple_diffusion_outputs/
```

## 🎯 Results Structure

After the sweep completes, you'll find:

### Downloaded Results (Local)

```
simple_diffusion/downloaded_images/
├── sweep_summary_YYYYMMDD_HHMMSS/
│   ├── README.md                    # Human-readable summary
│   ├── all_experiment_results.json  # Complete results
│   └── best_results_analysis.json   # Analysis of top performers
```

### On Server (Persistent)

```
~/models/
├── exp_0001_best_model.npz         # Best model from experiment 1
├── exp_0002_best_model.npz         # Best model from experiment 2
└── ...

~/logs/parameter_sweeps/
├── sweep_session_YYYYMMDD_HHMMSS/
│   ├── exp_0001/
│   │   ├── parameters.json
│   │   ├── training_history.json
│   │   └── results.json
│   └── ...
```

## ⚙️ Parameter Grid

The sweep explores:

- **Hidden layers:** 1, 2, 3, 4
- **Bottleneck sizes:** 16, 32, 56, 64, 96
- **Learning rates:** 0.001, 0.003, 0.01, 0.03
- **Leaky ReLU alpha:** 0.01, 0.05, 0.1, 0.2
- **Batch sizes:** 8, 16, 32
- **Gradient clipping:** 0.5, 1.0, 2.0

**Total combinations:** ~1,440 experiments

## 🔍 Monitoring

Monitor the training progress:

```bash
# Watch the logs in real-time
docker logs -f diffy-gpu-sweep

# Check GPU utilization
ssh user@your-server "nvidia-smi"
```

## 📈 Expected Performance

On RTX 5090:

- **~10-30 seconds** per experiment
- **100-200 experiments** per hour
- **2,000-4,000 experiments** in 20 hours
- **Automatic early stopping** for failed experiments

## 🛑 Early Termination

If you need to stop the sweep early:

```bash
# SSH to the server and stop the container
ssh user@your-server "docker stop diffy-gpu-sweep"
```

Results up to that point will still be downloaded!

## 🏆 Best Results

The system automatically identifies:

- **Lowest loss** configurations
- **Fastest converging** models
- **Most stable** training runs
- **Parameter sensitivity** analysis

## 🔧 Troubleshooting

### RTX 5090 Driver Issues

If you see CUDA errors, the RTX 5090 compatibility test will catch them:

```bash
# Test locally first
python3 nvidia/docker/test_rtx5090.py
```

### Out of Memory

Adjust batch sizes in `experiments/parameter_sweep.py`:

```python
'batch_size': [4, 8, 16],  # Reduce if OOM
```

### Slow Performance

Check if multiple experiments are getting the same data:

- Results should show varying loss patterns
- Check for data loading bottlenecks

## 🎯 Next Steps

After the sweep completes:

1. **Analyze** the downloaded results in `README.md`
2. **Load** the best model: `exp_XXXX_best_model.npz`
3. **Scale up** successful configurations for longer training
4. **Explore** parameter regions around the best results

The parameter sweep gives you the foundation to focus your research efforts on the most promising configurations! 🚀
