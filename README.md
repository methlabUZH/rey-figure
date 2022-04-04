# How to train and evaluate models

1. Prepare data (converting to grayscale, resizing and fusing the label files). This step assumes that your data is
   organized as follows:
    ```
    ├──data-root
        ├── DBdumps
        ├── ProlificExport
        ├── ReyFigures
        │ ├── data2018
        │ │ ├── newupload
        │ │ ├── newupload_15_11_2018
        │ │ ├── newupload_9_11_2018
        │ │ ├── uploadFinal
        │ │ └── uploadFinalREF
        │ └── data2021
        │     ├── KISPI
        │     ├── Tino_cropped
        │     ├── Typeform
        │     ├── USZ_fotos
        │     └── USZ_scans
        ├── UserRatingData
        └── simulated
    ```
   Run the following command
    ```python
    python prepare_data.py --data-root /path/to/data/ --dataset data-2018-2021 --preprocessing 0 --image-size 116 150
    ```
   which will create a new directory ``serialized-data/data-2018-2021-116x150-pp0`` in `data-root`, which contains the
   entire processed data and label files (train / test split).


3. Train models, e.g. multilabel classifier, via the following command
    ```python
    python train_multilabel.py --data-root /data-root/serialized-data/data-2018-2021-116x150-pp0 --results-dir /path/to/results --eval-test --id <id> --epochs 75 --batch-size 64 --lr 0.01 --gamma 0.95 --weighted-sampling 1 --image-size 116 150
    ```
   This will create a new dir in the specified results directory and save checkpoints + print outputs there.
   UPDATE: with hyperparameters.py and config.py one only has to run 
   ```
   python train_multilabel.py
   ``` 

4. To evaluate models, run the following command
   ```python
   python eval_multilabel.py --data-root /data-root/serialized-data/data-2018-2021-116x150-pp0 --results-dir /path/to/results --image-size 116 150 --batch-size 100 --tta --validation --angles -2.5 -1 0 1 2.5 
   ```
   UPDATE: with hyperparameters.py and config.py one only has to run 
   ```
   python eval_multilabel.py
   ``` 
   
   This will compute the overall MSE (note: change this to MAE in future), and create csv files which contains the
   prediction for each figure in the test set.

Note: the same workflow applies to training the regressor and for using different input sizes.