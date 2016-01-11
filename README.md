# regional

[![Latest Version](https://img.shields.io/pypi/v/regional.svg)](https://pypi.python.org/pypi/regional)
[![Build Status](https://img.shields.io/travis/freeman-lab/regional/master.svg)](https://travis-ci.org/freeman-lab/regional) 

simple manipulation and display of spatial regions in python

### install

```
pip install regional
```

### usage

####`region = one(coords)`

constructs a single region 

- `coords`
	- list of coordinates `[[x, y], [x, y], ...]`

![one](pngs/one.png)

####`regions = many(list)`

- `list` : 
	- list of regions `[region, region, ...]` or 
	- list of lists of coordinates `[[[x, y], [x, y], ...], [[x, y], [x, y], ...], ...]`

![many](pngs/many.png)

see the included [notebook](example.ipynb) for more examples

`one` region and `many` regions have the same attributes and methods, the only difference is that in the case of `many` regions they are just evaluated once per region

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

####`region.mask(dims, base, fill, stroke, value, cmap)`

generate image with regions as colored masks (`value` and `cmap` only for multiple regions)
