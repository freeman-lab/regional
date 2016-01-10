class one(object):
    
    def __init__(self, coordinates, values=None, id=None):
        self.coordinates = asarray(coordinates)

        if self.coordinates.ndim == 1 and len(self.coordinates) > 0:
            self.coordinates = asarray([self.coordinates])

        if id is not None:
            self.id = id
        else:
            self.id = 0

    def center(self):
        """
        Region center computed with a mean.
        """
        return mean(self.coordinates, axis=0)

    def polygon(self):
        """
        Bounding polygon as a convex hull.
        """
        from scipy.spatial import ConvexHull
        if len(self.coordinates) >= 4:
            inds = ConvexHull(self.coordinates).vertices
            return self.coordinates[inds]
        else:
            return self.coordinates

    def bbox(self):
        """
        Bounding box as minimum and maximum coordinates.
        """
        mn = amin(self.coordinates, axis=0)
        mx = amax(self.coordinates, axis=0)
        return concatenate((mn, mx))

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
        ntotal = float(len(set([tuple(x) for x in coords_self] 
            + [tuple(x) for x in coords_other])))

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
                + 1 + size * 2
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

