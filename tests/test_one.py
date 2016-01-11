from numpy import allclose
from regional import one

def test_construction():
	coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
	r = one(coords)
	assert allclose(r.coordinates, coords)
	

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
	assert allclose(r.coordinates, coords + [[1,1]])
	r = one(coords).merge(one([1, 1]))
	assert allclose(r.coordinates, coords + [[1,1]])


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
	other = one([[1, 1]])
	v = one(coords).overlap(other, 'fraction')
	assert v == 0.25





