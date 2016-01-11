# regional

simple manipulation and display of one or many spatial regions in python

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

### construction

####`region = one(coords)`

constructs a single region 

- `coords`
	- list of coordinates `[[x, y], [x, y], ...]`

####`regions = many(list)`

- `list` : 
	- list of regions `[region, region, ...]` or 
	- list of lists of coordinates `[[[x, y], [x, y], ...], [[x, y], [x, y], ...], ...]`

`one` region and `many` regions have the same attributes and methods, in the case of `many` regions they are just evaluated once per region

### attributes

####`region.hull`

convex hull

####`region.bbox`

rectangular bounding box

####`region.center`

euclidean center

####`region.extent`

total region extent

### methods

####`region.distance(other)`

distance to other region

####`region.merge(other)`

merge with other region

####`region.exclude(other)`

exclude other region

####`region.overlap(other, method)`

overlap with other region

####`region.crop(min, max)`

crop region to bounds

####`region.inbounds(min, max)`

check whether region falls completely within bounds

####`region.dilate(size)`

dilate region 

####`region.outline(inner, outer)`

compute region outline

####`region.mask(dims, base, fill, stroke)`

generate image with regions as colored masks
