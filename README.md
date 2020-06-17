
# Tabby Cat

Part of the [broadband.cat](http://broadband.cat) open source broadband modellin suite.


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
