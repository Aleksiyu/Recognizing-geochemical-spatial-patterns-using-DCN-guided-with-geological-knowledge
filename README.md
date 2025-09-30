# Recognizing-geochemical-spatial-patterns-using-DCN-guided-with-geological-knowledge
This study presents a model architecture that incorporates geological knowledge (GK) into a deformable convolutional network (DCN). By exploiting the inherent capability of deformable convolution kernels to adaptively modify sampling locations in response to input data characteristics, the model effectively learns and extracts irregular features within geochemical anomaly fields. Comparative analysis with traditional convolutional neural networks (CNNs) demonstrates that the proposed GK_DCN model exhibits enhanced flexibility and precision in capturing irregular features associated with complex mineralization processes. Furthermore, interpretability techniques such as Grad-CAM and offset visualization are employed to elucidate the model’s decision-making processes and to provide an intuitive representation of sampling point offsets, thereby revealing the learning mechanisms of DCNs in modeling irregular spatial features.

## Environment
This code was developed and tested in the following environment:
**Python**: 3.8  
**PyTorch**: 2.4.1  **TensorFlow**:2.4.0(CUDA 11.8+ recommended for GPU acceleration)  

## Requirements
- NumPy--1.19.5
- Pandas--1.1.5
- Matplotlib--3.5.1
- Scikit-learn--1.3.2
- OpenCV--4.10.0
- PyTorch--2.4.1
- torchvision--0.19.1
- TensorFlow--2.4.0

## File Structure & Functions
```
research/GK-DCN/
├── CNN.py                   #  Training and evaluating models of regular convolutional networks (CNNs)
├── DCN.py                   #  Training and evaluation models of Deformable Convolution (DCN)  
├── GK_CNN.py                #  CNN model constrained by integrated geological knowledge (GK)
├── GK_DCN.py                #  DCN model constrained by integrated geological knowledge (GK)
├── CNN_CAM.py               #  Perform Grad-CAM on CNN
├── DCN_CAM.py               #  Perform Grad-CAM visualization on DCN for model interpretation
├── DCN_Offset.py            #  Visualization script for the DCN model. Used to extract and plot the sampling point offset fields learned by the DCN.
```
