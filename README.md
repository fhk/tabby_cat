
# Tabby Cat

Part of the [broadband.cat](http://broadband.cat) open source broadband modellin suite.

# Run this

```
sudo apt-get install libspatialindex-dev 
git clone https://github.com/fhk/tabby_cat
conda create -n tabby_cat python=3.7
conda activate tabby_cat
conda install geopandas
pip install -r requirements.txt
python tabby_cat/main.py "State"
```

## DataLoader

- Pull data from OpenAddress and OpenStreetMap
- Create a data stream
- Create data layer

## Processor

- Connect demand points to base layer
- Convert geospatial locations to ids
- Output to filesystem

## Solver

- Load input
- Run model
- Output solution

## Deriver

- Join solver output to processor output to create geospatial representation


## Outputer

- Write geospatial solution output to filesystem
