# UQoL Quantification

**Environment:** Tested with PyTorch 1.8.1 and CUDA 10.2.

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n rdg
    source activate rdg
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 yacs openpyxl matplotlib pandas segmentation_models_pytorch albumentations
    ```

**Dataset:**

* Potsdam and Bhopal datasets are used for training.
* Download the dataset from: 

**Training:**

```bash
sh bashes/train_rdg.sh
```

3. **Acknowledgement**

* The code is based on [ResiDualGAN-DRDG](https://github.com/miemieyanga/ResiDualGAN-DRDG).

