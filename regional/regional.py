from numpy import asarray, amin, amax, sqrt, concatenate, mean, ndarray, sum, all, \
    ones, tile, expand_dims, zeros, where
import checkist

class one(object):
    
    def __init__(self, coordinates):
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
        if not isinstance(other, one):
            other = one(other)
        new = concatenate((self.coordinates, other.coordinates))
        return one(new)

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
        new = [c for c in self.coordinates if all(c >= min) and all(c < max)]
        return one(new)

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
        mincheck = sum(self.coordinates >= min, axis=1) == 0
        maxcheck = sum(self.coordinates < max, axis=1) == 0
        return True if (mincheck.sum() + maxcheck.sum()) == 0 else False

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
            'fraction' (symmetric) or 'rates' (asymmetric)
        """
        checkist.opts(method, ['fraction', 'rates'])

        coords_self = self.coordinates.tolist()
        coords_other = other.coordinates.tolist()

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
            from skimage.morphology import binary_dilation

            size = (size * 2) + 1
            coords = self.coordinates
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)]
            extent += 1 + size * 2
            m = zeros(extent)
            coords = (coords - self.bbox[0:len(self.center)] + size)
            m[coords.T.tolist()] = 1
            m = binary_dilation(m, ones((size, size)))
            new = asarray(where(m)).T + self.bbox[0:len(self.center)] - size
            new = [c for c in new if all(c >= 0)]
        else:
            return self
        
        return one(new)

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
        if isinstance(other, list) or isinstance(other, ndarray):
            other = asarray(other)
            coords_other = asarray(where(other)).T.tolist()
        else:
            coords_other = other.coordinates.tolist()

        coords_self = self.coordinates.tolist()

        complement = [a for a in coords_self if a not in coords_other]

        return one(complement)

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

    def mask(self, dims=None, base=None, fill='deeppink', stroke=None):
        """
        Create a mask image with colored regions.

        Parameters
        ----------
        dims : tuple, optional, default = None
            Dimensions of embedding image,
            will be ignored if background image is provided.

        base : array-like or str, optional, default = None
            Background, can provide a string color specifier, 
            RGB values, or a 2d or 3d array.

        fill : str or array-like, optional, default = 'pink'
            String color specifier, or RGB values

        stroke : str or array-like, optional, default = None
            String color specifier, or RGB values
        """
        from matplotlib import colors

        fill = color2array(fill)
        stroke = color2array(stroke)
        base = color2array(base)

        if dims is None or (base is not None and asarray(base).shape == (3,)):
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)] + 1
            offset = self.bbox[0:len(self.center)]
        else:
            extent = dims
            offset = [0, 0]

        if base is None:
            base = ones(tuple(extent) + (3,))
            
        else:
            base = asarray(base)
            if base.shape == (3,):
                m = zeros(tuple(extent) + (3,))
                for channel in range(3):
                    m[:,:,channel] = base[channel]
                base = m
            elif base.ndim < 3:
                base = tile(expand_dims(base,2),[1,1,3])
                offset = [0, 0]
            else:
                offset = [0, 0]

        for channel in range(3):
            inds = asarray([[c[0], c[1], channel] for c in self.coordinates - offset])
            base[inds.T.tolist()] = fill[channel]

        if stroke is not None:
            mn = [0, 0]
            mx = [base.shape[0], base.shape[1]]
            edge = self.outline(0, 1).coordinates - offset
            edge = [e for e in edge if all(e >= mn) and all(e < mx)]
            if len(edge) > 0:
                for channel in range(3):
                    inds = asarray([[c[0], c[1], channel] for c in edge])
                    base[inds.T.tolist()] = stroke[channel]

        return base

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
        elif isinstance(regions, list):
            self.regions = []
            for r in regions:
                self.regions.append(one(r))
        else:
            raise Exception("Input type not recognized, must be region, list of regions, "
                            "or list of coordinates, got %s" % type(regions))

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self.regions[selection]
        else:
            self.regions = self.regions[selection]
            return self

    def combiner(self, prop):
        return [getattr(r, prop) for r in self.regions]

    def evaluator(self, func, *args, **kwargs):
        return [getattr(r, func)(*args, **kwargs) for r in self.regions]

    def updater(self, func, *args, **kwargs):
        return many([getattr(r, func)(*args, **kwargs) for r in self.regions])

    @property
    def center(self):
        """
        Region center computed with a mean.
        """
        return self.combiner('center')

    @property
    def coordinates(self):
        return self.combiner('coordinates')

    @property
    def hull(self):
        """
        Bounding convex hull as a polygon.
        """
        return self.combiner('hull')

    @property
    def bbox(self):
        """
        Bounding box as minimum and maximum coordinates.
        """
        return self.combiner('bbox')
    
    @property
    def area(self):
        """
        Region area as number of pixels.
        """
        return self.combiner('area')

    @property
    def count(self):
        """
        Number of regions
        """
        return len(self.regions)

    def distance(self, other):
        return self.evaluator('distance', other)

    def overlap(self, other, method='fraction'):
        return self.evaluator('overlap', other, method)

    def inbounds(self, min, max):
        return self.evaluator('inbounds', min, max)

    def merge(self, other):
        return self.updater('merge', other)

    def exclude(self, other):
        return self.updater('exclude', other)

    def crop(self, min, max):
        return self.updater('crop', min, max)

    def dilate(self, size):
        return self.updater('dilate', size)

    def outline(self, inner, outer):
        return self.updater('outline', inner, outer)

    def mask(self, dims=None, base=None, fill='deeppink', stroke=None):
        """
        Create a mask image with colored regions.

        Parameters
        ----------
        dims : tuple, optional, default = None
            Dimensions of embedding image,
            will be ignored if background image is provided.

        base : array-like, optional, default = None
            Array to use as base background image.

        fill : str or array-like, optional, default = 'pink'
            String color specifier, or RGB values

        stroke : str or array-like, optional, default = None
            String color specifier, or RGB values
        """
        from matplotlib import colors

        base = color2array(base)

        if dims is None or (base is not None and not asarray(base).shape == (3,)):
            mins = asarray([b[0:2] for b in self.bbox])
            maxes = asarray([b[2:] for b in self.bbox])
            extent = maxes.max(axis=0) - mins.min(axis=0) + 1
        else:
            extent = dims

        if base is None:
            base = ones(tuple(extent) + (3,))
        else:
            base = asarray(base)
            if base.shape == (3,):
                m = zeros(tuple(extent) + (3,))
                for channel in range(3):
                    m[:,:,channel] = base[channel]
                base = m
            elif base.ndim < 3:
                base = tile(expand_dims(base,2),[1,1,3])

        for r in self.regions:
            base = r.mask(base=base, fill=fill, stroke=stroke)

        return base

    def __repr__(self):
        s = 'regions'
        s += '\ncount: %g' % self.count
        return s
    
keys = ['distance', 'merge', 'exclude', 'overlap', 'crop',
        'inbounds', 'dilate', 'outline']
        
for k in keys:
    many.__dict__[k].__doc__ = one.__dict__[k].__doc__

def color2array(name):
    if isinstance(name, str):
        from matplotlib import colors
        return asarray(colors.hex2color(colors.cnames[name]))
    else:
        return name

