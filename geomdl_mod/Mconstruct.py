"""
.. module:: construct
    :platform: Unix, Windows
    :synopsis: Provides functions for constructing and extracting spline geometries

.. moduleauthor:: Onur Rauf Bingol <orbingol@gmail.com>

"""

from geomdl import shortcuts
from geomdl import knotvector
from geomdl import compatibility
from geomdl.exceptions import GeomdlException


def construct_surface(direction, args, **kwargs):
    """ Generates surfaces from curves.

    Keyword Arguments (optional):
        * ``degree``: degree of the 2nd parametric direction
        * ``knotvector``: knot vector of the 2nd parametric direction
        * ``rational``: flag to generate rational surfaces

    :param direction: the direction that the input curves_v lies, i.e. u or v
    :type direction: str
    :param args: a list or tuple of curves_v
    :type args: Union[list, tuple]
    :return: Surface constructed from the curves_v on the given parametric direction
    """
    # Input validation
    possible_dirs = ['u', 'v']
    if direction not in possible_dirs:
        raise GeomdlException("Possible direction values: " + ", ".join([val for val in possible_dirs]),
                              data=dict(input_dir=direction))

    size_other = len(args)
    if size_other < 2:
        raise GeomdlException("You need to input at least 2 curves_v")

    # Get keyword arguments
    degree_other = kwargs.get('degree', 2)
    knotvector_other = kwargs.get('knotvector', knotvector.generate(degree_other, size_other))
    rational = kwargs.get('rational', args[0].rational)

    # Construct the control points of the new surface
    degree = args[0].degree
    num_ctrlpts = args[0].ctrlpts_size
    new_ctrlpts = []
    new_weights = []
    for idx, arg in enumerate(args):
        if degree != arg.degree:
            raise GeomdlException("Input curves_v must have the same degrees",
                                  data=dict(idx=idx, degree=degree, degree_arg=arg.degree))
        if num_ctrlpts != arg.ctrlpts_size:
            raise GeomdlException("Input curves_v must have the same number of control points",
                                  data=dict(idx=idx, size=num_ctrlpts, size_arg=arg.ctrlpts_size))
        new_ctrlpts += list(arg.ctrlpts)
        if rational:
            if arg.weights is None:
                raise GeomdlException("Expecting a rational curve",
                                      data=dict(idx=idx, rational=rational, rational_arg=arg.rational))
            new_weights += list(arg.weights)

    # Set variables w.r.t. input direction
    if direction == 'u':
        degree_u = degree_other
        degree_v = degree
        knotvector_u = knotvector_other
        knotvector_v = args[0].knotvector
        size_u = size_other
        size_v = num_ctrlpts
    else:
        degree_u = degree
        degree_v = degree_other
        knotvector_u = args[0].knotvector
        knotvector_v = knotvector_other
        size_u = num_ctrlpts
        size_v = size_other
        if rational:
            ctrlptsw = compatibility.combine_ctrlpts_weights(new_ctrlpts, new_weights)
            ctrlptsw = compatibility.flip_ctrlpts_u(ctrlptsw, size_u, size_v)
            new_ctrlpts, new_weights = compatibility.separate_ctrlpts_weights(ctrlptsw)
        else:
            new_ctrlpts = compatibility.flip_ctrlpts_u(new_ctrlpts, size_u, size_v)

    # Generate the surface
    ns = shortcuts.generate_surface(rational)
    ns.degree_u = degree_u
    ns.degree_v = degree_v
    ns.ctrlpts_size_u = size_u
    ns.ctrlpts_size_v = size_v
    ns.ctrlpts = new_ctrlpts
    if rational:
        ns.weights = new_weights
    ns.knotvector_u = knotvector_u
    ns.knotvector_v = knotvector_v

    # Return constructed surface
    return ns


def extract_curves(psurf, **kwargs):
    """ Extracts curves_v from a surface.

    The return value is a ``dict`` object containing the following keys:

    * ``u``: the curves_v which generate u-direction (or which lie on the v-direction)
    * ``v``: the curves_v which generate v-direction (or which lie on the u-direction)

    As an example; if a curve lies on the u-direction, then its knotvector is equal to surface's knotvector on the
    v-direction and vice versa.

    The curve extraction process can be controlled via ``extract_u`` and ``extract_v`` boolean keyword arguments.

    :param psurf: input surface
    :type psurf: abstract.Surface
    :return: extracted curves_v
    :rtype: dict
    """
    if psurf.pdimension != 2:
        raise GeomdlException("The input should be a spline surface")
    if len(psurf) != 1:
        raise GeomdlException("Can only operate on single spline surfaces")

    # Get keyword arguments
    extract_u = kwargs.get('extract_u', True)
    extract_v = kwargs.get('extract_v', True)

    # Get data_v from the surface object
    surf_data = psurf.data_v
    rational = surf_data['rational']
    degree_u = surf_data['degree'][0]
    degree_v = surf_data['degree'][1]
    kv_u = surf_data['knotvector'][0]
    kv_v = surf_data['knotvector'][1]
    size_u = surf_data['size'][0]
    size_v = surf_data['size'][1]
    cpts = surf_data['control_points']

    # Determine object type
    obj = shortcuts.generate_curve(rational)

    # v-direction
    crvlist_v = []
    if extract_v:
        for u in range(size_u):
            curve = obj.__class__()
            curve.degree = degree_v
            curve.set_ctrlpts([cpts[v + (size_v * u)] for v in range(size_v)])
            curve.knotvector = kv_v
            crvlist_v.append(curve)

    # u-direction
    crvlist_u = []
    if extract_u:
        for v in range(size_v):
            curve = obj.__class__()
            curve.degree = degree_u
            curve.set_ctrlpts([cpts[v + (size_v * u)] for u in range(size_u)])
            curve.knotvector = kv_u
            crvlist_u.append(curve)

    # Return shapes as a dict object
    return dict(u=crvlist_u, v=crvlist_v)
