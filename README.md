# Emotion Recognition Using Smartwatch Sensors: From Deep Learning to Lightweight Models

## Project Overview

This project explores emotion recognition using 7-channel smartwatch sensor data through a complete workflow: from advanced deep learning model development on a source dataset to practical transfer learning implementation. The key finding is that lightweight models perform competitively with complex deep learning architectures, making real-world deployment more feasible.

## Complete Workflow

### Phase 1: Original Dataset Model Development (Files 0-7)

**Dataset**: 7-channel accelerometer (x,y,z), gyroscope (x,y,z), and heart rate data
**Goal**: Develop optimal emotion recognition models

#### Model Development Pipeline
- **File 0**: Data channel analysis and structure examination
- **File 1**: Deep learning model experimentation (notebook)
- **File 2**: Data augmentation experiments  
- **Files 3**: TCN pipeline development (basic, improved, spatial variants)
- **File 4**: Three-TCN ensemble implementation
- **File 5**: Hybrid ensemble optimization
- **File 6**: Stacking ensemble (final best approach)
- **File 7**: Model pipeline documentation

#### Key Results
- **Individual Models**: Spatial TCN achieved highest single-model performance
- **Best Ensemble**: Stacking ensemble (3 TCNs + 2 ML models) reached 78.24% accuracy
- **Architecture Finding**: TCN outperformed other deep learning approaches
- **Ensemble Strategy**: Combining neural networks with traditional ML improved performance

### Phase 2: Transfer Learning to EmoWear Dataset (Files a-e)

**Dataset**: EmoWear - real-world emotion recognition data from 48 participants

**Goal**: Test transferability and develop practical lightweight models

#### Transfer Learning Pipeline
- **File a**: EmoWear dataset analysis and structure validation
- **File b**: Data preprocessing to match original 7-channel format
- **File c**: Wrist-focused 7-channel mapping (5 wrist + 2 chest sensors)
- **File d**: Improved TCN transfer learning with regularization
- **File e**: Final hybrid ensemble (TCN + traditional ML models)

#### Key Findings
- **Transfer Learning**: Basic TCN achieved 46.09% accuracy (vs 44.74% from scratch)
- **Model Complexity**: Advanced architectures performed worse than simple ones
- **Best Result**: Lightweight ensemble (basic TCN + 3 ML models) achieved 50.15%
- **Practical Insight**: Traditional ML competitive with deep learning approaches

## Performance Summary

### Original Dataset (High-Performance Track)
- Basic TCN: 70.79%
- Improved TCN: 70.11% 
- Spatial TCN: 72.77%
- Three-TCN Ensemble: 76.26%
- **Stacking Ensemble: 78.24%** (best performance)

### Transfer Learning (EmoWear Dataset)
- Transfer Learning (Frozen): 45.04%
- Transfer Learning (Fine-tune): 46.17%
- From Scratch: 44.74%
- **Lightweight Ensemble: 50.15%** (final best)

## Key Technical Insights

### Model Complexity vs Performance
The project revealed a crucial finding: **simpler models often outperform complex architectures** in practical scenarios.

**What Works**:
- Basic TCN architectures
- Traditional ML (Random Forest, Logistic Regression)  
- Ensemble of simple models
- Minimal preprocessing
- Standard training procedures

**What Doesn't Work**:
- Complex attention mechanisms
- Heavy signal filtering
- Advanced architectural features
- Curriculum learning strategies
- Extensive data augmentation

### Practical Implications
1. **Lightweight Deployment**: Simple models enable real-world smartwatch implementation
2. **Traditional ML Relevance**: Classical machine learning remains highly competitive
3. **Transfer Learning Viability**: Models successfully transfer across datasets
4. **Ensemble Benefits**: Combining diverse simple models beats single complex ones

## Technical Architecture

### Final Lightweight Ensemble (50.15% accuracy)
```
Ensemble Components:
├── Basic TCN (Neural Network)           # 30% weight
├── Random Forest (Traditional ML)       # 50% weight  
├── Logistic Regression (Traditional ML) # 10% weight
└── Linear SVM (Traditional ML)          # 10% weight
```

### Data Pipeline
- **Input**: 7-channel sensor data (accelerometer, gyroscope, heart rate)
- **Preprocessing**: Savitzky-Golay filtering, MinMax normalization
- **Windowing**: 128 samples (4 seconds) with 75% overlap
- **Output**: 3-class emotion classification (Sad, Neutral, Happy)

## Dataset Details

### Original Dataset (juancq)
- **Channels**: 7 (3-axis accelerometer + 3-axis gyroscope + heart rate)
- **Performance**: Up to 78.24% with advanced ensembles
- **Use Case**: Model development and architecture optimization

### EmoWear Dataset  
- **Participants**: 48 individuals
- **Samples**: 102,523 total windows
- **Channels**: 7 (5 wrist sensors + 2 chest sensors mapped to match original)
- **Performance**: 50.15% with lightweight ensemble
- **Use Case**: Transfer learning and practical deployment validation

Note: Due to size constraints, the following data is not included in the repository:
- Model files (*.pth, *.pkl, *.h5, *.hdf5)
- Preprocessed data files (*.npy, *.npz)
- EmoWear dataset
- Results and comparison files
- Generated visualizations (*.png, *.jpg, *.pdf)

## Key Files

### Model Development (0-7)
- `3_final_tcn_pipeline.py`: Basic TCN implementation
- `3_final_improved_tcn_pipeline.py`: Enhanced TCN with more parameters
- `3_final_spatial_tcn_pipeline.py`: Spatial attention TCN
- `4_three_tcn_ensemble.py`: Three-model ensemble
- `6_stacking_ensemble.py`: Final stacking approach

### Transfer Learning (a-e)
- `a_emowear_analysis.py`: Dataset structure analysis
- `b_emowear_preprocessing.py`: Data preprocessing pipeline
- `c_emowear_7channel_data_preprocessing.py`: Channel mapping
- `d_improved_tcn_transfer.py`: Improved TCN transfer learning
- `e_hybrid_ensemble_current_tcn.py`: Final hybrid ensemble

### Documentation
- `7_Model_Pipeline_Guide.md`: Complete technical documentation
- `f_EMOWEAR_COMPLETE_SUMMARY.md`: Detailed experimental results

## Scientific Contribution

This work demonstrates that:
1. **Lightweight models are practical**: Simple architectures can achieve competitive performance
2. **Transfer learning works**: Models successfully adapt across different sensor configurations
3. **Traditional ML remains relevant**: Classical algorithms complement deep learning effectively
4. **Ensemble diversity matters**: Multiple simple models outperform single complex ones
5. **Real-world deployment is feasible**: Performance levels suitable for practical applications

## Applications

The lightweight nature of the final models makes them suitable for:
- **Smartwatch Applications**: Real-time emotion monitoring
- **Mobile Health**: Continuous wellness tracking  
- **Edge Computing**: On-device processing without cloud dependency
- **Battery Efficiency**: Reduced computational requirements
- **Scalable Deployment**: Easy implementation across diverse hardware

## Conclusion

While deep learning achieves impressive results on the source dataset (78.24%), the transfer learning experiments reveal that lightweight models (50.15%) offer a practical alternative for real-world deployment. The competitive performance of traditional ML approaches suggests that the computational overhead of complex deep learning may not always be justified, particularly for resource-constrained applications like smartwatch-based emotion recognition.

This finding has positive implications for practical deployment: simpler models mean longer battery life, faster inference, and easier implementation on edge devices.