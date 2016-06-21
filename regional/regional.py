from numpy import asarray, amin, amax, sqrt, concatenate, arange, \
    mean, ndarray, sum, all, ones, tile, expand_dims, zeros, where, integer
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

    @property
    def extent(self):
        """
        Total region extent.
        """
        return self.bbox[2:] - self.bbox[0:2] + 1
    
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
        unique = set([tuple(x) for x in new.tolist()])
        final = asarray([list(x) for x in unique])
        return one(final)

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
            from scipy.ndimage.morphology import binary_dilation
            size = (size * 2) + 1
            coords = self.coordinates
            tmp = zeros(self.extent + size * 2)
            coords = (coords - self.bbox[0:len(self.center)] + size)
            tmp[coords.T.tolist()] = 1
            tmp = binary_dilation(tmp, ones((size, size)))
            new = asarray(where(tmp)).T + self.bbox[0:len(self.center)] - size
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

    def mask(self, dims=None, base=None, fill='deeppink', stroke='black', background=None):
        """
        Create a mask image with colored regions.

        Parameters
        ----------
        dims : tuple, optional, default = None
            Dimensions of embedding image,
            will be ignored if background image is provided.

        base : array-like, optional, default = None
            Base image, can provide a 2d or 3d array, 
            if unspecified will be white.

        fill : str or array-like, optional, default = 'pink'
            String color specifier, or RGB value

        stroke : str or array-like, optional, default = None
            String color specifier, or RGB value

        background : str or array-like, optional, default = None
            String color specifier, or RGB value
        """
        fill = getcolor(fill)
        stroke = getcolor(stroke)
        background = getcolor(background)

        if dims is None and base is None:
            region = one(self.coordinates - self.bbox[0:2])
        else:
            region = self

        base = getbase(base=base, dims=dims, extent=self.extent, background=background)

        if fill is not None:
            for channel in range(3):
                inds = asarray([[c[0], c[1], channel] for c in region.coordinates])
                base[inds.T.tolist()] = fill[channel]

        if stroke is not None:
            mn = [0, 0]
            mx = [base.shape[0], base.shape[1]]
            edge = region.outline(0, 1).coordinates
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
        if isinstance(selection, int) or isinstance(selection, integer):
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
    def extent(self):
        """
        Total region extent.
        """
        return self.combiner('extent')
    
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

    def mask(self, dims=None, base=None, fill='deeppink', stroke='black', background=None, 
             cmap=None, cmap_stroke=None, value=None):
        """
        Create a mask image with colored regions.

        Parameters
        ----------
        dims : tuple, optional, default = None
            Dimensions of embedding image,
            will be ignored if background image is provided.

        base : array-like, optional, default = None
            Array to use as base image, can be 2d (BW) or 3d (RGB).

        fill : str or array-like, optional, default = 'pink'
            String color specifier, or RGB values,
            or a list of either.

        stroke : str or array-like, optional, default = None
            String color specifier, or RGB values,
            or a list of either.

        background : str or array-like, optional, default = None
            String color specifier, or RGB values.

        cmap : str or colormap, optional, deafult = None
            String specifier for colormap, or colormap. Will control
            both fill and stroke. Use cmap_stroke to
            set stroke independently.

        cmap_stroke : str or colormap, optional, deafult = None
            String specifier for colormap, or colormap, for stroke only.

        value : array-like, optional, default = None
            Value per region for use with colormap.
            If None and cmap is specified, will use the range
            from 0 to the number of regions.
        """
        if (cmap is not None or cmap_stroke is not None) and value is None:
            value = arange(self.count)
        background = getcolor(background)
        stroke = getcolors(stroke, self.count, cmap_stroke, value)
        fill = getcolors(fill, self.count, cmap, value)

        minbound = asarray([b[0:2] for b in self.bbox]).min(axis=0)
        maxbound = asarray([b[2:] for b in self.bbox]).max(axis=0)
        extent = maxbound - minbound + 1

        if dims is None and base is None:
            regions = [one(r.coordinates - minbound) for r in self.regions]
        else:
            regions = self.regions

        base = getbase(base=base, dims=dims, extent=extent, background=background)

        for i, r in enumerate(regions):
            f = fill[i] if fill is not None else None
            s = stroke[i] if stroke is not None else None
            base = r.mask(base=base, fill=f, stroke=s)

        return base

    def __repr__(self):
        s = 'regions'
        s += '\ncount: %g' % self.count
        return s
    
keys = ['distance', 'merge', 'exclude', 'overlap', 'crop',
        'inbounds', 'dilate', 'outline']
        
for k in keys:
    many.__dict__[k].__doc__ = one.__dict__[k].__doc__

def getcolor(spec):
    """
    Turn optional color string spec into an array.
    """
    if isinstance(spec, str):
        from matplotlib import colors
        return asarray(colors.hex2color(colors.cnames[spec]))
    else:
        return spec

def getcolors(spec, n, cmap=None, value=None):
    """
    Turn list of color specs into list of arrays.
    """
    if cmap is not None and spec is not None:
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import get_cmap
        if isinstance(cmap, LinearSegmentedColormap):
            return cmap(value)[:, 0:3]
        if isinstance(cmap, str):
            return get_cmap(cmap, n)(value)[:, 0:3]
    if isinstance(spec, str):
        return [getcolor(spec) for i in range(n)]
    elif isinstance(spec, list) and isinstance(spec[0], str):
        return [getcolor(s) for s in spec]
    elif (isinstance(spec, list) or isinstance(spec, ndarray)) and asarray(spec).shape == (3,):
        return [spec for i in range(n)]
    else:
        return spec

def getbase(base=None, dims=None, extent=None, background=None):
    """
    Construct a base array from optional arguments.
    """
    if dims is not None:
        extent = dims
    if base is None and background is None:
        return ones(tuple(extent) + (3,))
    elif base is None and background is not None:
        base = zeros(tuple(extent) + (3,))
        for channel in range(3):
            base[:, :, channel] = background[channel]
        return base
    elif base is not None and base.ndim < 3:
        return tile(expand_dims(base, 2),[1, 1, 3])
    else:
        return base