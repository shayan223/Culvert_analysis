# DINO DETR Implementation for Culvert Analysis

**Introduction:**
DINO (DETR with Improvised Denoising Anchor Boxes for end-to-end Object Detection) is a state-of-the-art object detection transformer model used to analyze the distinctive topographic patterns of culverts in LiDAR DEMs dataset. 

**Requirements:**

- [Python 3.7.3 or above](https://www.python.org/downloads)
- [PyTorch 1.9.0 or above](https://pytorch.org/get-started/locally)
- [CUDA 11.1 or above](https://developer.nvidia.com/cuda-downloads)

**DataSet:**
You can find the big_geo_data dataset in the `main` branch of this repository.

**Process:**

1. Clone the repository.
2. Unpack the data from the `main` branch.
3. Configure the backbone model:

   - For DINO 4 convscale:
     ```
     python engine_DINO.py -c config/DINO_4scale.py -m "cuda"
     ```

   - For DINO 4 convscale with Swin Transformer:
     ```
     python engine_DINO.py -c config/DINO_4scale_swin.py -m "cuda"
     ```

4. Perform validation or inference on validation data for DINO models:

   - For DINO 4 convscale, check the path of the saved model in `inference.py` and run:
     ```
     python validate_dino.py -c config/DINO_4scale.py -m "cuda"
     ```

   - For DINO 4 convscale with Swin Transformer, check the path of the saved model in `inference.py` and run:
     ```
     python validate_dino.py -c config/DINO_4scale_swin.py -m "cuda"
     ```

Note: The `engine_DINO.py` contains code to load the DINO architecture.

**Results:**

The results of the dino implementation for culvert analysis can be viewed in the `./Results` directory. 

