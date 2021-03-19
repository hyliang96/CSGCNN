# Experiment on VOC Part

## Dir structure

The dir structure is

```
./
    requirements.txt  python environment requirements
    starting.sh       download dataset, checkpoints, and set __result__/ link.

    train/       training models
    VOC/         dataset preprocess and loading
    gradmap/     evaluating with gradient map
    activamap/   evaluating with cam/filter activation map
    mutualinfo/  evaluating with mutual information score

    checkpoints/ checkpoints in our paper
        CSG.pt   training with our CSG method
        STD.pt   training with standard method
    __data__/    -- soft link --> dataset dir
    __result__/  -- soft link --> result dir, saving experiment logs, checkpoints
        VOCPart_128x128_pretrained/
            <experiment-id>/
                <seed>/
                    checkpoints/
                    events.out.tfevents.xxx
                    log

```

## Device

GPU Memory requirement: >=5000MB

## Environment

* Apply the python environment:

    ```bash
    virtualenv -p python3 venv  # we use Python 3.7.3
    source venv/bin/activate venv
    pip install -r requirements.txt
    ```

* Download checkpoints, download preprocessed dataset, and set result dir:

    ```bash
    bash ./starting.sh
    ```

    This will

    * download the checkpoints reported in our paper to `checkpoints/{STD,CSG}.pt`
    * download the preprocessed dataset VOCPart used in our papaer to `preprocessed_VOC_Part/` and set `__data__` linked to it
    * make a dir to save trianing result and set `__result__` linked to it


## Datasetsss

### Preprocessed dataset

The datatset we used is VOC Part dataset, which is preproceed from Pascal VOC 2010 dataset. To *reproduce* our paper, please download and use the preprocessed dataset with `bash ./starting.sh`. Then you will get the preprocessed VOC Part dataset as below:

    ```
    ./
        __data__/ --> preprocessed_VOC_Part/
        preprocessed_VOC_Part/
            processed/
                128x128/
                    obj_img/
                    obj_mask/
                    part_mask/
                metadata/
                    train.txt
                    val.txt
    ```

### Preprocessing by yoursef

If you want to preprocess Pascal VOC 2010 dataset and get VOC Part by yourself, you can run the commands below. Notice that the VOC Part dataset preprocessed by yourself might be slightly different from the aforementioned "preprocessed dataset" used in our paper, because we fix some bugs in preprocessing codes in this version.

* Download Pascal VOC 2010 dataset dataset:

    ```bash
    bash VOC/download.sh
    ```

* Then preprocess the dataset with so as to create `__data__/processed/`:

    ```bash
    python VOC/preprocess.py
    ```

* Then you will get a dataset dir as below:

    ```
    ./
        __data__/ --> <dataset-dir>
            raw/
                VOC2010/    # traininig and validation data
                metadata/   # Pascal VOC part annotation
            processed/
                metadata/   # the division of training and validation set
                128x128/    # processed images are 128x128 pixel
                    obj_img/       # cropped and resized images for each object instance
                    obj_mask/      # cropped and resized segmentation ground truth for each object instance
                    part_mask/     # cropped and resized segmentation ground truth for each part in an object instance
    ```

## Training

To *approximately* reproduce the training process in our paper, please run the code below

* Set the "training settings" in  `train/run.py` as following, which is the default setting.

    ```python
    gpu_ids = '0'
    ifmask = True
    train = True
    ```

    set `ifmask = True` and `ifmask = False` to train CSG/STD CNN respectively.

* Run the following codes to train a CSG/STD CNN. It will report the classification accuracy on testing set.

    ```python
    python train/run.py
    ```

## Evaluation

Run the commands below to reproduce the evaluation on the mutual information score and the localization performance in our paper. This will evaluate the checkpoints we used in our paper, i.e. `./checkpoints/{STD,CSG}.pt`.

* MIS (mutual information score) between classes and channels

    ```bash
    python mutualinfo/mutualinfo.py
    ```

* Localization performance of GradMap (gradient map)

    ```bash
    python gradmap/gradmap.py
    ```

* Localization performance of ActivMap (activaiton map) and CAM (classification activation map)

    ```bash
    python activmap/activmap.py
    ```
