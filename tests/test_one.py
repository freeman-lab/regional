from numpy import allclose
from regional import one

def test_construction():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = one(coords)
	assert allclose(r.coordinates, coords)
	coords = [1, 1]
	r = one(coords)
	assert allclose(r.coordinates, [coords])
	

def test_center():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = one(coords)
	assert allclose(r.center, [0.5, 0.5])


def test_hull():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	r = one(coords)
	assert allclose(r.hull, [[0, 0], [2, 0], [2, 2], [0, 2]])


def test_bbox():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	r = one(coords)
	assert allclose(r.bbox, [0, 0, 2, 2])


def test_extent():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	r = one(coords)
	assert allclose(r.extent, [3, 3])


def test_area():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = one(coords)
	assert r.area == 4


def test_distance():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = one(coords)
	assert r.distance([1, 1]) == 0


def test_merge():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = one(coords).merge([1, 1])
	assert equal_sets(r.coordinates.tolist(), coords + [[1, 1]]) 
	r = one(coords).merge(one([1, 1]))
	assert equal_sets(r.coordinates.tolist(), coords + [[1, 1]]) 


def test_merge_nonunique():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = one(coords).merge([[1, 1], [0, 0]])
	assert equal_sets(r.coordinates.tolist(), coords + [[1,1]])
	r = one(coords).merge(one([[1, 1], [0, 0]]))
	assert equal_sets(r.coordinates.tolist(), coords + [[1,1]])


def test_crop():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = one(coords).crop([0, 0], [1, 1])
	assert allclose(r.coordinates, [[0, 0]])


def test_inbounds():
	coords = [[1, 1], [1, 2], [2, 1], [2, 2]]
	v = one(coords).inbounds([0, 0], [3, 3])
	assert v == True
	v = one(coords).inbounds([0, 0], [2, 2])
	assert v == False
	v = one(coords).inbounds([1, 1], [3, 3])
	assert v == True


def test_overlap():
	coords = [[1, 1], [1, 2], [2, 1], [2, 2]]
	v = one(coords).overlap(one([1, 1]), 'fraction')
	assert v == 0.25
	v = one(coords).overlap(one([[1, 1],[1, 2]]), 'fraction')
	assert v == 0.5
	v = one(coords).overlap(one([1, 1]), 'rates')
	assert v == (0.25, 1.0)
	v = one(coords).overlap(one([[1, 1], [1, 2], [3, 3], [4, 4]]), 'rates')
	assert v == (0.5, 0.5)


def test_dilate():
	v = one([1, 1]).dilate(1)
	assert allclose(v.coordinates, [[0, 0], [0, 1], [0, 2], [1, 0], 
		[1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
	v = one([1, 1]).dilate(0)
	assert allclose(v.coordinates, [[1, 1]])


def test_exclude():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = one(coords).exclude(one([[0, 0], [0, 1]]))
	assert allclose(r.coordinates, [[1, 0], [1, 1]])
	r = one(coords).exclude([[0, 0], [0, 1]])
	assert allclose(r.coordinates, [[0, 0], [0, 1], [1, 0]])


def test_outline():
	coords = [[1, 1]]
	r = one(coords).outline(0, 1)
	assert allclose(r.coordinates, [[0, 0], [0, 1], [0, 2], 
		[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]])


def test_mask():
	coords = [[1, 1]]
	r = one(coords)
	im = r.mask(fill='red')
	assert im.shape == (1, 1, 3)
	assert allclose(im, [[[1, 0, 0]]])
	im = r.mask(fill=[1, 0, 0])
	assert im.shape == (1, 1, 3)
	assert allclose(im, [[[1, 0, 0]]])


def test_mask_no_fill():
	coords = [[1, 1]]
	r = one(coords)
	im = r.mask(fill=None, stroke='black', dims=(2,2))
	assert allclose(im[:,:,0], [[0, 0], [0, 1]])
	assert allclose(im[:,:,1], [[0, 0], [0, 1]])
	assert allclose(im[:,:,2], [[0, 0], [0, 1]])


def test_mask_colors():
	coords = [[1, 1]]
	r = one(coords)
	im = r.mask(fill='blue', stroke=None, dims=(2, 2))
	assert im.shape == (2, 2, 3)
	assert allclose(im[:, :, 0], [[1, 1], [1, 0]])
	assert allclose(im[:, :, 1], [[1, 1], [1, 0]])
	assert allclose(im[:, :, 2], [[1, 1], [1, 1]])
	im = r.mask(fill='red', stroke='black', dims=(2, 2))
	assert im.shape == (2, 2, 3)
	assert allclose(im[:, :, 0], [[0, 0], [0, 1]])
	assert allclose(im[:, :, 1], [[0, 0], [0, 0]])
	assert allclose(im[:, :, 2], [[0, 0], [0, 0]])
	im = r.mask(fill='red', stroke=None, background='blue', dims=(2, 2))
	assert im.shape == (2, 2, 3)
	assert allclose(im[:, :, 0], [[0, 0], [0, 1]])
	assert allclose(im[:, :, 1], [[0, 0], [0, 0]])
	assert allclose(im[:, :, 2], [[1, 1], [1, 0]])


def equal_sets(a, b):
	aset = set([tuple(x) for x in a])
	bset = set([tuple(x) for x in b])
	return aset == bset