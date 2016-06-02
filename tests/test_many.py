from numpy import allclose, int64
from regional import one, many

def test_construction():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = many([one(coords), one(coords)])
	assert r.count == 2
	assert allclose(r.coordinates, [coords, coords])
	r = many([coords, coords])
	assert r.count == 2
	assert allclose(r.coordinates, [coords, coords])
	r = many([coords, coords, coords])
	assert r.count == 3
	assert allclose(r.coordinates, [coords, coords, coords])


def test_index():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = many([one(coords), one(coords)])
	assert allclose(r[0].coordinates, coords)
	assert allclose(r[int64(0)].coordinates, coords)


def test_center():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = many([coords, coords])
	assert allclose(r.center, [[0.5, 0.5], [0.5, 0.5]])


def test_hull():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	truth = [[0, 0], [2, 0], [2, 2], [0, 2]]
	r = many([coords, coords])
	assert allclose(r.hull, [truth, truth])


def test_bbox():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	truth = [0, 0, 2, 2]
	r = many([coords, coords])
	assert allclose(r.bbox, [truth, truth])


def test_extent():
	coords = [[0, 0], [0, 2], [2, 0], [1, 1], [2, 2]]
	r = many([coords, coords])
	assert allclose(r.extent, [[3, 3], [3, 3]])


def test_area():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = many([coords, coords])
	assert r.area == [4, 4]


def test_distance():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	r = many([coords, coords])
	assert r.distance([1, 1]) == [0, 0]


def test_merge():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	truth = coords + [[1,1]]
	r = many([coords, coords]).merge([1, 1])
	assert allclose(r.coordinates, [truth, truth])


def test_crop():
	coords = [[0, 0], [0, 2], [2, 0], [2, 2]]
	truth = [[0, 0]]
	r = many([coords, coords]).crop([0, 0], [1, 1])
	assert allclose(r.coordinates, [truth, truth])


def test_inbounds():
	coords = [[1, 1], [1, 2], [2, 1], [2, 2]]
	v = many([coords, coords]).inbounds([0, 0], [3, 3])
	assert v == [True, True]


def test_overlap():
	coords = [[1, 1], [1, 2], [2, 1], [2, 2]]
	v = many([coords, coords]).overlap(one([1, 1]))
	assert v == [0.25, 0.25]


def test_dilate():
	v = many([one([1, 1]), one([1, 1])]).dilate(1)
	truth = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
	assert allclose(v.coordinates, [truth, truth])


def test_exclude():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	truth = [[1, 0], [1, 1]]
	r = many([coords, coords]).exclude(one([[0, 0], [0, 1]]))
	assert allclose(r.coordinates, [truth, truth])


def test_outline():
	coords = [[1, 1]]
	truth = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
	r = many([coords, coords]).outline(0, 1)
	assert allclose(r.coordinates, [truth, truth])


def test_mask():
	r = many([one([0, 0]), one([1, 1])])
	im = r.mask(fill='red', stroke=None)
	assert allclose(im[:,:,0], [[1, 1], [1, 1]])
	assert allclose(im[:,:,1], [[0, 1], [1, 0]])
	assert allclose(im[:,:,2], [[0, 1], [1, 0]])
	im = r.mask(fill=[1, 0, 0], stroke=None)
	assert allclose(im[:,:,0], [[1, 1], [1, 1]])
	assert allclose(im[:,:,1], [[0, 1], [1, 0]])
	assert allclose(im[:,:,2], [[0, 1], [1, 0]])


def test_mask_background():
	r = many([one([0, 0]), one([1, 1])])
	im = r.mask(fill='red', stroke=None, background='black')
	assert allclose(im[:,:,0], [[1, 0], [0, 1]])
	assert allclose(im[:,:,1], [[0, 0], [0, 0]])
	assert allclose(im[:,:,2], [[0, 0], [0, 0]])
	im = r.mask(fill=[1, 0, 0], stroke=None, background='black')
	assert allclose(im[:,:,0], [[1, 0], [0, 1]])
	assert allclose(im[:,:,1], [[0, 0], [0, 0]])
	assert allclose(im[:,:,2], [[0, 0], [0, 0]])


def test_mask_colors():
	r = many([one([0, 0]), one([1, 1])])
	im = r.mask(fill=['red','blue'], stroke=None, background='black')
	assert allclose(im[:,:,0], [[1, 0], [0, 0]])
	assert allclose(im[:,:,1], [[0, 0], [0, 0]])
	assert allclose(im[:,:,2], [[0, 0], [0, 1]])
	im = r.mask(fill=[[1, 0, 0], [0, 0, 1]], stroke=None, background='black')
	assert allclose(im[:,:,0], [[1, 0], [0, 0]])
	assert allclose(im[:,:,1], [[0, 0], [0, 0]])
	assert allclose(im[:,:,2], [[0, 0], [0, 1]])


def test_mask_colormap():
	r = many([one([0, 0]), one([1, 1])])
	im = r.mask(cmap='gray', stroke=None, value=[0, 1], background='red')
	assert allclose(im[:,:,0], [[0, 1], [1, 1]])
	assert allclose(im[:,:,1], [[0, 0], [0, 1]])
	assert allclose(im[:,:,2], [[0, 0], [0, 1]])

