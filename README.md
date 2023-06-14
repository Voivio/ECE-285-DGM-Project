# ECE-285-DGM-Project

Course project repo for ECE 285 - Deep Generative Models.

Topic: 3D object generation with diffusion-based models.

## Setup

### Environment

After cloning this repo, you can create the virtual environmant with the following commands:

```bash
conda env create -f environment.yml
onda activate ddpm
```

### Dataset

The [ShapeNet](https://shapenet.org/) dataset can be downloaded from Google Drive using the link [here](https://drive.google.com/file/d/1I7iinoG8K64U27inCy7Lk7R0ZDsd21r7/view?usp=sharing) and [here](https://drive.google.com/file/d/1WOcTkrTDONrGvizN3TNyuRlkGp9OlX7-/view?usp=sharing).

### Checkpoints

The trained checkpoints can be downloaded from Google Drive using the link [here](https://drive.google.com/file/d/1BrUgcX-hSmF-Px8uof1lv4dgfLsJnWpc/view?usp=sharing).

## Running the code

### Training from scratch

To train the model from scratch, the following command starts training:

```bash
python train_gen.py \
    --dataset_path=<PATH TO YOUR DOWNLOADED shapenet.hdf5> \
    --max_iters=5000 \
    --val_freq=50 \
    --model='flow' \
    --encoder='pointnet' \
    --diffusion_layer_type='squash' \
    --categories="airplane,bag,basket,bathtub,bed,bench,bottle,bowl,bus,cabinet" \
    --device="cuda" \
    --log_root="./logs_gen"
```

The previous command trains the baseline model. To alter options, you may:

+ `model`: "gaussian" trains a model with a Gaussian prior distribution.
+ `encoder`: "resnet" trains a model using the ResNet encoder.
+ `diffusion_layer_type`: "non-squash" trains a model using the concatenated linear layers.
+ `categories`: "airplane" trains a model using a subset of dataset with only the airplane category.

The checkpoints and logs will be stored in the path specified with `log_root`.

### Testing with checkpoints

To test the trained model, the following command starts evaluation:

```bash
python test_gen.py \
    --ckpt=<PATH TO TRAINED CHECKPOINT> \
    --dataset_path=<PATH TO YOUR DOWNLOADED shapenet.hdf5> \
    --categories="airplane" \
    --save_dir=<PATH TO THE FOLDER FOR AVING RESULTS> \
    --device="cuda" \
    --batch_size=32
```

The previous command tests the baseline model. To alter options, you may:

+ `ckpy`: the path can lead to the provided checkpoints, or the checkpoints stored during training.
+ `categories`: should match the `categories` argument used in trianing.
+ `batch_size`: reduce the batch size if out-of-memoty error occurs.

To further visuailze the generated point clouds, please replace the `point_file_paths` to your paths of output as specified by the `save_dir` argument in the third code block from `visualization.ipynb` and run the third code block (first/second blocks need to be runned first).

```python
point_file_paths = OrderedDict({
    "airplane": "/data/dongyin/logs_gen/results/GEN_Ours_car_1686692959/out.npy",
    "car": "/data/dongyin/logs_gen/results/GEN_Ours_airplane_1686692525/out.npy"
})
```

**Note**: The evaluation takes a long time. To avoid long computation, please uncomment line 12-23 and comment line 25-101 in file `evaluation/evaluation_metrics.py` to avoid calculating EMD-related metrics.

## Reference:

1. Deep Generative Models on 3D Representations: A Survey. [Paper](https://arxiv.org/pdf/2210.15663.pdf) and [repo](https://github.com/justimyhxu/awesome-3d-generation).
2. Diffusion Probabilistic Models for 3D Point Cloud Generation. [Paper](https://arxiv.org/abs/2103.01458) and [repo](https://github.com/luost26/diffusion-point-cloud#diffusion-probabilistic-models-for-3d-point-cloud-generation). 
3. LION: Latent Point Diffusion Models for 3D Shape Generation.[Paper](https://arxiv.org/abs/2210.06978) and [repo](https://github.com/nv-tlabs/LION).
4. GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images.[Paper](https://nv-tlabs.github.io/GET3D/assets/paper.pdf) and [repo](https://github.com/nv-tlabs/GET3D).
