# Diffy Experiment Log

## Experiment #1: Baseline 56D Bottleneck Test

**Date**: Initial run  
**Duration**: ~5200 epochs (stopped early)  
**Architecture**: 128D → 56D bottleneck → 128D

### Results

- **Colors Learned**: 14/264 (5.3% accuracy)
- **Challenge**: 264 colors in 56D space (0.21 dimensions per color)
- **Data Type**: Random 128D vectors (pure memorization task)

### Key Observations

1. **Extreme bottleneck confirmed brutal**: 0.21D per color is insufficient
2. **Random vectors are hard to memorize**: No patterns to exploit
3. **Network struggles significantly**: Only 5.3% success rate
4. **Validates "1D per item" hypothesis**: 56D clearly insufficient for 264 items

### Conclusions

- 56D bottleneck cannot effectively memorize 264 arbitrary vectors
- Need either: larger bottleneck OR fewer items OR structured data
- Current architecture tests fundamental memorization limits

---

## New Experimental Strategy: 5K Epoch Iterations

### Why 5K Cycles?

- **Faster iteration** on parameter changes
- **Better A/B testing** of architectural modifications
- **Quicker feedback** on what's working
- **More experiments** in same time frame

### Parameters to Test (5K epochs each):

1. **Bottleneck size**: 56D → 64D → 80D → 100D
2. **Learning rate**: 0.00001 → 0.0001 → 0.001
3. **Batch size**: 2 → 4 → 8 → 16
4. **Network depth**: 3 layers → 4 layers → 5 layers
5. **Activation**: Leaky ReLU α (0.01 → 0.1 → 0.2)

### Success Metrics (at 5K epochs):

- **Baseline**: 14 colors (5.3%) - our reference point
- **Improvement goal**: >20 colors (>7.5%)
- **Strong improvement**: >40 colors (>15%)

### Data Collection Plan:

- Document final accuracy for each 5K run
- Track which parameter changes help/hurt
- Build systematic understanding of what affects memorization capacity

---

## Next Steps:

1. Add CSV logging for data tracking
2. Implement model checkpointing
3. Start systematic 5K epoch parameter sweeps
4. Document results in this log
