# WiREDiff: a Wind Resolution-Enhancing Diffusion Model
___


To access the processed machine-learning wind speed dataset used for this exploration, visit the Caltech Data Respository page: [raw data](https://data.caltech.edu/records/czs3p-5ss80)

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```

## N-Con*ffusion*

Fill the pretrained model weights into the finetuning config file. For training N-Con*ffusion* on the task of super-resolution, run:

```
python3 sr_finetune_bounds.py -p calibration -c config/finetune_bounds_16_128_conffusion.json --enable_wandb --finetune_loss quantile_regression
```

Once the finetuning is complete, we can test the finetuned bounds. For this, replace the value of `EXPERIMENT_NAME` 
of the `bounds_resume_state` argument in the file `test_finetune_bounds_16_128_conffusion.json` 
with the actual experiment name of the finetuned run (i.e. have it point to the dir with the saved checkpoints).

Finally, run the following command 
```
python3 test_finetuned_bounds.py --enable_wandb
```


## Weights and Biases

The library now supports experiment tracking, model checkpointing and model prediction visualization with [Weights and Biases](https://wandb.ai/site). You will need to [install W&B](https://pypi.org/project/wandb/) and login by using your [access token](https://wandb.ai/authorize). 

```
pip install wandb

# get your access token from wandb.ai/authorize
wandb login
```

W&B logging functionality is added to the `sr.py`, `sample.py` and `infer.py` files. You can pass `-enable_wandb` to start logging.

- `-log_wandb_ckpt`: Pass this argument along with `-enable_wandb` to save model checkpoints as [W&B Artifacts](https://docs.wandb.ai/guides/artifacts). Both `sr.py` and `sample.py` is enabled with model checkpointing. 
- `-log_eval`: Pass this argument along with `-enable_wandb` to save the evaluation result as interactive [W&B Tables](https://docs.wandb.ai/guides/data-vis). Note that only `sr.py` is enabled with this feature. If you run `sample.py` in eval mode, the generated images will automatically be logged as image media panel. 
- `-log_infer`: While running `infer.py` pass this argument along with `-enable_wandb` to log the inference results as interactive W&B Tables. 

You can find more on using these features [here](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/pull/44)


## Acknowledgments
- The SR3 implementation is based on <a href="https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement" target="_blank">this</a> unofficial implementation of SR3. 
- The implementation of the calibration and evaluation metrics is based on <a href="https://github.com/aangelopoulos/im2im-uq" target="_blank">this</a> official implementation of im2im-uq.
- The implementation of N-Con*ffusion* is based on <a href="https://github.com/eliahuhorwitz/Conffusion" target="_blank">this</a>
