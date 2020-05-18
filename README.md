# HW 1 CMLM - URBANSOUND

In order to run the project copy the directory `UrbanSound8K` in the root directory

# Docs
Docs are in `html\lib\index.html`

# Run
Simply run the `main.py` by executing
```python main.py```

# Documentation
Full Report available into  `Group 18 - Homework 1 Paper.pdf`

Generate documentation running:
```pdoc --html lib --force```


```
mount -t tmpfs -o rw,size=20G tmpfs /mnt/ramdisk
cd /mnt/ramdisk
wget wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar xzf UrbanSound8K.tar.gz
git clone https://github.com/minettiandrea/hw1-cmls-urbansound.git
cd hw1-cmls-urbansound
mv ../UrbanSound8K ./
```
