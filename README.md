# ```trafficlightdetector.py``` Usage

## Imports
```python
import cv2
from glob import glob
from trafficlightdetector import TrafficLightDetector
```

## Define color bounds
```python
red_bound = ([0, 151, 178], [71, 215, 359])
green_bound = ([72, 217, 160], [200, 255, 359])
# color_bounds = {'red':red_bound, 'green':green_bound}
color_bounds = {'green':green_bound}
```

## Load reference images
```python
reference_images = []
reference_paths = glob('./reference/*.jpg')
for path in reference_paths:
    reference_images.append(cv2.imread(path))
```

## Initialize ```TrafficLightDetector``` object
```python
TLD = TrafficLightDetector(reference_images, color_bounds, color_threshold=0.1, feature_threshold=24, display=True)
```

## Detect state
**Single image**
```python
query = cv2.imread('query.jpg')
state = TLD.get_state(query)
```

```
>> print(state)
>> red
```

**Video stream from webcam**
```python
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    query = frame
    try:
        state = TLD.get_state(query)
    except:
        state = 'ERROR'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
*Note: the ```try``` ```except``` is there to handle a currently mysterious error that occasionally pops up*

Output:

```
>> None
```
or
```
>> green
```
or
```
>> ERROR
```