from numpy import asarray, amin, amax, sqrt, concatenate, mean, ndarray

class one(object):
    
    def __init__(self, coordinates, values=None, id=None):
        self.coordinates = asarray(coordinates)

        if self.coordinates.ndim == 1 and len(self.coordinates) > 0:
            self.coordinates = asarray([self.coordinates])

        if id is not None:
            self.id = id
        else:
            self.id = 0

    @property
    def center(self):
        """
        Region center computed with a mean.
        """
        return mean(self.coordinates, axis=0)

    @property
    def hull(self):
        """
        Bounding polygon as a convex hull.
        """
        from scipy.spatial import ConvexHull
        if len(self.coordinates) >= 4:
            inds = ConvexHull(self.coordinates).vertices
            return self.coordinates[inds]
        else:
            return self.coordinates

    @property
    def bbox(self):
        """
        Bounding box as minimum and maximum coordinates.
        """
        mn = amin(self.coordinates, axis=0)
        mx = amax(self.coordinates, axis=0)
        return concatenate((mn, mx))

    @property
    def area(self):
        """
        Region area as number of pixels.
        """
        return len(self.coordinates)

    def distance(self, other):
        """
        Distance between the center of this region and another.

        Parameters
        ----------
        other : one region, or array-like
            Either another region, or the center of another region.
        """
        from numpy.linalg import norm
        if isinstance(other, one):
            other = other.center
        return norm(self.center - asarray(other), ord=2)
        
    def merge(self, other):
        """
        Combine this region with other.
        """
        self.coordinates = concatenate((self.coordinates, other.coordinates))

        if hasattr(self, 'values'):
            self.values = concatenate((self.values, other.values))

        return self

    def crop(self, min, max):
        """
        Crop a region by removing coordinates outside bounds.

        Follows normal slice indexing conventions.

        Parameters
        ----------
        min : tuple
            Minimum or starting bounds for each axis.

        max : tuple
            Maximum or ending bounds for each axis.
        """
        self.coordinates =  [c for c in coords if all(c >= minBound) and all(c < maxBound)]
        return self

    def inbounds(self, min, max):
        """
        Check if a region falls entirely inside bounds.

        Parameters
        ----------
        min : tuple
            Minimum bound to check for each axis.

        max : tuple
            Maximum bound to check for each axis.
        """
        mincheck = sum(self.coordinates < min, axis=1) == 0
        maxcheck = sum(self.coordinates > max, axis=1) == 0
        return True if (mincheck + maxcheck) == 0 else False

    def overlap(self, other, method='fraction'):
        """
        Compute the overlap between this region and another.

        Optional methods are a symmetric measure of overlap based on the fraction
        of intersecting pixels relative to the union ('fraction'), 
        or an assymmetric measure of overlap using precision and recall 
        rates ('rates').

        Parameters
        ----------
        other : one region
            The region to compute overlap with.

        method : str
            Which estimate of overlap to compute, options are
            'fraction' (symmetric) 'rates' (asymmetric) or 'correlation'
        """
        checkist.opts(method, ['fraction', 'rates'])

        coords_self = aslist(self.coordinates)
        coords_other = aslist(other.coordinates)

        intersection = [a for a in coords_self if a in coords_other]
        nhit = float(len(intersection))
        ntotal = float(len(set([tuple(x) for x in coords_self] + 
            [tuple(x) for x in coords_other])))

        if method == 'rates':
            recall = nhit / len(coords_self)
            precision = nhit / len(coords_other)
            return recall, precision

        if method == 'fraction':
            return nhit / float(ntotal)

    def dilate(self, size):
        """
        Dilate a region using morphological operators.

        Parameters
        ----------
        size : int
            Size of dilation in pixels
        """
        if size > 0:
            size = (size * 2) + 1
            from skimage.morphology import binary_dilation

            coords = self.coordinates
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)]
            extent += 1 + size * 2
            m = zeros(extent)
            coords = (coords - self.bbox[0:len(self.center)] + size)
            m[coords.T.tolist()] = 1
            m = binary_dilation(m, ones((size, size)))
            newcoords = asarray(where(m)).T + self.bbox[0:len(self.center)] - size
            newcoords = [c for c in newcoords if all(c >= 0)]
            self.coordinates = newcoords
        
        return self

    def exclude(self, other):
        """
        Remove coordinates from another region or an array.

        If other is an array, will remove coordinates of all
        non-zero elements from this region. If other is a region,
        will remove any matching coordinates.

        Parameters
        ----------
        other : ndarray or one region
            Region to remove.
        """
        if isinstance(other, ndarray):
            coords_other = asarray(where(other)).T
        else:
            coords_other = aslist(other.coordinates)

        coords_self = aslist(self.coordinates)

        complement = [a for a in coords_self if a not in coords_other]
        self.coordinates = complement

        return self

    def outline(self, inner, outer):
        """
        Compute region outline by differencing two dilations.

        Parameters
        ----------
        inner : int
            Size of inner outline boundary (in pixels)

        outer : int
            Size of outer outline boundary (in pixels)
        """
        return self.dilate(outer).exclude(self.dilate(inner))

    def __repr__(self):
        s = 'region'
        for opt in ['center', 'bbox']:
            o = self.__getattribute__(opt)
            os = o.tolist() if isinstance(o, ndarray) else o
            s += '\n%s: %s' % (opt, repr(os))
        return s

class many(object):

    def __init__(self, regions):
        if isinstance(regions, one):
            self.regions = [regions]
        elif isinstance(regions, list) and isinstance(regions[0], one):
            self.regions = regions
        elif isinstance(sources, list):
            self.regions = []
            for r in regions:
                self.regions.append(one(r))
        else:
            raise Exception("Input type not recognized, must be region, list of regions, "
                            "or list of coordinates, got %s" % type(sources))

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self.regions[selection]
        else:
            self.regions = self.regions[selection]
            return self

    def combiner(self, prop):
        return [getattr(r, prop) for r in self.regions]

    def evaluator(self, prop):
        return [getattr(r, prop) for r in self.regions]

    @property
    def center(self):
        return self.combiner('center')

    @property
    def coordinates(self):
        return self.combiner('coordinates')

    @property
    def hull(self):
        return self.combiner('hull')

    @property
    def area(self):
        return self.combiner('area')

    @property
    def count(self):
        """
        Number of regions
        """
        return len(self.regions)

    
    
    def __repr__(self):
        s = 'regions'
        s += '\ncount: %g' % self.count
        return s
    