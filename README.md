# ```trafficlightdetector.py``` Usage

## Imports
```python
import cv2
from glob import glob
from trafficlightdetector import TrafficLightDetector
```

## Define ```color_bounds```
```python
red_bound = ([0,0,0], [255,255,360])
green_bound = ([0,0,0], [255,255,360])
color_bounds = {'red':red_bound, 'green':green_bound}
```
*NOTE: replace above HSV color values*

## Load ```reference_images```
```python
reference_images = []
reference_paths = glob('./reference/*.jpg')
for path in reference_paths:
    reference_images.append(cv2.imread(path))
```
*NOTE: place reference images in folder named "reference"*

## Initialize ```TrafficLightDetector``` object
```python
TLD = TrafficLightDetector(reference_images, color_bounds, color_threshold=32, feature_threshold=10)
```

## Detect state
```python
query = cv2.imread('query.jpg')
state = TLD.get_state(query)
```

```
>> print(state)
>> red
```