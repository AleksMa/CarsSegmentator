### Cars segmentation into 5 classes

Based on [Kaggle task](https://www.kaggle.com/datasets/intelecai/car-segmentation)

#### Usage

```
pip3 install -r requirements.txt
python3 main.py
```

You may enable/disable generation of extra samples using `GENERATE_EXTRA_SAMPLES` constant in config.py.  
If you need to delete extra samples, use command
```
ls | grep -E ".*(flipped|transformed).*" | xargs rm
```

#### Results

Results are located in `./src/test/results` dir.
