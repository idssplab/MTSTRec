# MTSTRec: Multimodal Time-Aligned Shared Token Recommender 
The **Multimodal Time-Aligned Shared Token Recommender (MTSTRec)** is a transformer-based model designed for sequential recommendation in e-commerce. Unlike traditional methods that fuse multimodal data (e.g., text, images, and pricing) early or late, MTSTRec introduces a time-aligned shared token for each product. This token enables efficient cross-modal fusion while maintaining temporal alignment in users’ browsing sequences. By capturing rich, modality-specific features and aligning them in time, MTSTRec offers a more comprehensive understanding of user preferences. Experiments show that MTSTRec outperforms existing multimodal approaches, setting new benchmarks for sequential recommendation performance.

## Model Architecture 
<div align="center">
   <img src= "https://github.com/idssplab/MTSTRec/blob/main/MTSTRec.png" width="80%" height="80%">
</div>


## Install Necessary Packages
```
pip install -r requirements.txt
```


## Download the Dataset
You can download our released datasets (Food E-commerce and House-Hold E-commerce) at [here](https://drive.google.com/file/d/1H4I_8H-DpiCh7vSu9ScCmOik7RwlWUXz/view?usp=sharing).


## Dataset preprocess
The `Preprocessing_hm` folder contains the data preprocessing code for the H&M (Trousers) dataset.


## File Description
<!-- #region -->
| File                 | Description                                                                             |
| ---------------------| --------------------------------------------------------------------------------------- |
| util.py              | Useful data structures for log management, input file processing and evaluation.        |
| parameters.py        | Parameter settings for traning and evaluation.                                          |
| dataloader.py        | Handles preprocessing and management of datasets for training and evaluation.           | 
| module.py            | Implements the transformer encoder for sequence modeling and feature extraction.        |
| model.py             | Defines the MTSTRec model architecture used for training and inference.                  |
| trainer.py           | Contains the training pipeline, including model training, validation.                    |
| mainfinal.py         | The script to execute the end-to-end training and evaluation process.                   |
| run.py               | The script to feed customized parameters into mainfinal.py for process.                  |
| inference.py         | The script to execute the end-to-end evaluation process.                                |
| ./logs               | Folder for saving log files.                                                              |
| ./models             | Folder for putting checkpoints.                                                         |
<!-- #endregion -->


## Model training
```python
python3 run.py
```


## Model Inference
* Parameters can be set in parameters.py
```python
python3 inference.py --use_token --use_style --use_text --use_price
```