# regional

simple manipulation and display of spatial regions in python

### install

```
pip install regional
```

### example

```
from regional import one
from showit import image

region = one([[0, 0], [0, 1], [1, 1], [1, 0]])
image(region.mask())
```

### usage

####`region = one(coords)`

constructs a single region 

- `coords` : list of coordinates in the form `[[x, y], [x, y], ...]`

#### attributes

####`region.polygon()`

####`region.center()`

####`region.bbox()`

#### methods

####`region.distance(other)`

####`region.merge(other)`

####`region.exclude(other)`

####`region.overlap(other, method)`

####`region.crop(min, max)`

####`region.dilate(size)`

####`region.outline(inner, outer)`

####`territory = many(coords)`

constructs a collection of regions

- `coords` : a list of regions, or a list of lists of coordinates in the form `[[[x, y], [x, y], ...], [[x, y], [x, y], ...], ...]`

