
"""
** Import parameters that used in model establishment.
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from visualization import *
from odbAccess import *
import functions.loading_protocols as lp
import functions.structural_components as sc
import functions.set_selection as ss
import math


def param_decoder(member):
    """
    Decode parameters.
    :param member: beam or column member.
    :return: decoded params.
    """
    return member.d, member.w, member.tf, member.tw, member.r, member.mem_len, member.len_beam, member.len_solid, \
           member.mesh_size_solid, member.mesh_size_web, member.mesh_size_flange, member.num_flange_up, \
           member.num_flange_low, member.num_web, member.num_mid, member.num_side, member.num_half_arc, member.beam_ele_num, \
           member.len_beam / member.beam_ele_num, member.step_period, member.axial_load


def member_creator(component_type, section_name, mem_len, material_list, loading_protocol, main_folder,
                   axial_load_ratio, support_ratio, boundary, support_ratio_limit=50.):
    """
    Create a member object.
    :param component_type: component type, choose between "Beam" and "Column".
    :param section_name: input section name.
    :param mem_len: member total length.
    :param material_list: material list containing E, s_{y,0}, Q_{\infty}, b, D_{\infty}, a, C_1, \gamma_1, C_2, \gamma_2.
    :param loading_protocol: loading protocol, choose among 'Symmetric', 'Collapse_consistent' and 'Monotonic'.
    :param main_folder: main folder's absolute path.
    :param axial_load_ratio: axial load to yield strength ratio (only for column).
    :param support_ratio: beam's maximum unbraced length to section weak-axis gyration radius ratio (only for beam).
    :param boundary: boundary condition, choose between 'Fixed' and 'Flexible'.
    :param support_ratio_limit: beam's required support ratio limit.
    :return: beam or column member object.
    """
    # Warning messages
    if loading_protocol not in ['Symmetric', 'Collapse_consistent', 'Monotonic']:
        raise ValueError("Loading protocol is not correct.")
    if len(material_list) != 10:
        raise ValueError("Incomplete material information!")
    if boundary not in ['Fixed', 'Flexible']:
        raise ValueError("Boundary is not correct.")
    # Define member object
    if component_type == 'Column':
        if support_ratio is not None:
            raise ValueError('Columns should have no supports!')
        if axial_load_ratio <= 0.:
            raise ValueError("Columns should have compressive axial load!")
        member = sc.Column(section_name, mem_len, material_list, loading_protocol, axial_load_ratio, boundary, main_folder)
    elif component_type == "Beam":
        if axial_load_ratio != 0.:
            raise ValueError('Axial load should be zero for beams!')
        if boundary is not 'Fixed':
            raise ValueError('Beams should only have fixed ends!')
        if support_ratio is None or support_ratio <= 0.:
            raise ValueError('Beams should have support!')
        member = sc.Beam(section_name, mem_len, material_list, loading_protocol, support_ratio, support_ratio_limit, main_folder)
    else:
        raise NameError("The component type should be Column or Beam!")
    return member


def build_amplitude(member, loading_protocol, model_name="Model-Macro"):
    """
    Establish amplitude based on loading protocol.
    :param member: beam or column member.
    :param loading_protocol: loading protocol, choose among 'Symmetric', 'Collapse_consistent' and 'Monotonic'.
    :param model_name: model name.
    """
    # Half beam loading protocols
    if member.type == "Half_Beam":
        symmetric_lp = [i / 2. for i in lp.Symmetric_LP]
        collapse_consistent_lp = [i / 2. for i in lp.Collapse_consistent_LP]
        monotonic_lp = [i / 2. for i in lp.Monotonic_LP]
    # Full beam and Column loading protocols
    elif member.type == "Full_Beam" or member.type == "Column":
        symmetric_lp = lp.Symmetric_LP
        collapse_consistent_lp = lp.Collapse_consistent_LP
        monotonic_lp = lp.Monotonic_LP
    else:
        raise ValueError("Member type error!")
    # Amplitude if symmetric load
    if loading_protocol == "Symmetric":
        step_period = len(symmetric_lp) - 1
        mdb.models[model_name].TabularAmplitude(data=tuple(enumerate(symmetric_lp)), name='Amp1')
    # Amplitude if monotonic load
    elif loading_protocol == "Monotonic":
        step_period = len(monotonic_lp) - 1
        mdb.models[model_name].TabularAmplitude(data=tuple(enumerate(monotonic_lp)), name='Amp1')
    # Amplitude if collapse consistent load
    elif loading_protocol == "Collapse_consistent":
        step_period = len(collapse_consistent_lp) - 1
        mdb.models[model_name].TabularAmplitude(data=tuple(enumerate(collapse_consistent_lp)), name='Amp1')
    else:
        raise ValueError("Please input correct loading protocol!")
    member.step_period = int(step_period)
    # Build user-defined amplitude
    mdb.models[model_name].UserAmplitude(name='Amp2', numVariables=3, timeSpan=STEP)


def part_creator(member, model_name="Model-Macro"):
    """
    Create solid and beam element parts.
    :param member: member object.
    :param model_name: model name.
    :return: unmeshed solid element and beam element parts.
    """
    # Read geometrical and mesh parameters
    d, w, tf, tw, r, _, len_beam, len_solid, _, _, _, _, _, _, _, _, _, _, _, _, _ = param_decoder(member)
    mat_prop_beam, mat_prop_solid = member.mat_prop_beam, member.mat_prop_solid
    trans_shear_stiffness = member.trans_shear_stiffness
    # Create solid I-section column sketch
    sketch1 = mdb.models[model_name].ConstrainedSketch(name='Profile1', sheetSize=2000)
    sketch1.Line(point1=(w/2, d/2), point2=(-w/2, d/2))
    sketch1.Line(point1=(w/2, d/2), point2=(w/2, d/2-tf))
    sketch1.Line(point1=(-w/2, d/2-tf), point2=(-w/2, d/2))
    sketch1.Line(point1=(tw/2+r, d/2-tf), point2=(w/2, d/2-tf))
    sketch1.Line(point1=(-tw/2-r, d/2-tf), point2=(-w/2, d/2-tf))
    sketch1.Line(point1=(tw/2, d/2-tf-r), point2=(tw/2, -d/2+tf+r))
    sketch1.Line(point1=(-tw/2, d/2-tf-r), point2=(-tw/2, -d/2+tf+r))
    sketch1.Line(point1=(-tw/2-r, -d/2+tf), point2=(-w/2, -d/2+tf))
    sketch1.Line(point1=(tw/2+r, -d/2+tf), point2=(w/2, -d/2+tf))
    sketch1.Line(point1=(w/2, -d/2+tf), point2=(w/2, -d/2))
    sketch1.Line(point1=(-w/2, -d/2+tf), point2=(-w/2, -d/2))
    sketch1.Line(point1=(-w/2, -d/2), point2=(w/2, -d/2))
    sketch1.ArcByCenterEnds(center=(tw/2+r, d/2-tf-r), point1=(tw/2+r, d/2-tf), point2=(tw/2, d/2-tf-r))
    sketch1.ArcByCenterEnds(center=(tw/2+r, -d/2+tf+r), point1=(tw/2, -d/2+tf+r), point2=(tw/2+r, -d/2+tf))
    sketch1.ArcByCenterEnds(center=(-tw/2-r, d/2-tf-r), point1=(-tw/2, d/2-tf-r), point2=(-tw/2-r, d/2-tf))
    sketch1.ArcByCenterEnds(center=(-tw/2-r, -d/2+tf+r), point1=(-tw/2-r, -d/2+tf), point2=(-tw/2, -d/2+tf+r))
    # Create solid I-section column part
    part1 = mdb.models[model_name].Part(dimensionality=THREE_D, name='Part-Solid', type=DEFORMABLE_BODY)
    part1.BaseSolidExtrude(depth=len_solid, sketch=sketch1)
    # Create beam sketch
    sketch2 = mdb.models[model_name].ConstrainedSketch(name='Profile2', sheetSize=5000)
    sketch2.Line(point1=(-len_beam/2, 0.), point2=(len_beam/2, 0.))
    # Create beam I-section column part
    part2 = mdb.models[model_name].Part(dimensionality=THREE_D, name='Part-Beam', type=DEFORMABLE_BODY)
    part2.BaseWire(sketch=sketch2)
    # Material definition
    # Create material property for solid
    material1 = mdb.models[model_name].Material(name='M-SOLID')
    material1.UserMaterial(type=MECHANICAL, mechanicalConstants=mat_prop_solid)
    material1.Depvar(n=19)
    # Create material property for beam
    material2 = mdb.models[model_name].Material(name='M-BEAM')
    material2.UserMaterial(type=MECHANICAL, mechanicalConstants=mat_prop_beam)
    material2.Depvar(n=3)
    # Assign sections for solid and beam
    mdb.models[model_name].HomogeneousSolidSection(name='Section-Solid', material='M-SOLID')
    mdb.models[model_name].IProfile(name='i_profile', l=d/2, h=d, b1=w, b2=w, t1=tf, t2=tf, t3=tw)
    section2 = mdb.models[model_name].BeamSection(name='Section-Beam', material='M-BEAM', profile='i_profile',
                                                  integration=DURING_ANALYSIS, poissonRatio=0.3)
    section2.TransverseShearBeam(scfDefinition=ANALYSIS_DEFAULT, k23=trans_shear_stiffness, k13=trans_shear_stiffness)
    part1.SectionAssignment(region=regionToolset.Region(cells=part1.cells), sectionName='Section-Solid')
    part2.SectionAssignment(region=regionToolset.Region(edges=part2.edges), sectionName='Section-Beam')
    # Assign beam orientation
    part2.assignBeamSectionOrientation(region=regionToolset.Region(edges=part2.edges),
                                       method=N1_COSINES, n1=(0., 0., -1.))
    return part1, part2


def create_mesh(member, part1, part2):
    """
    Create mesh on solid and beam parts.
    :param member: beam or column member.
    :param part1: solid element segment part.
    :param part2: beam element segment part.
    :return: meshed solid and beam parts.
    """
    # Read geometrical and mesh parameters
    d, w, tf, tw, r, _, _, len_solid, mesh_size_solid, mesh_size_web, mesh_size_flange, num_flange_up, \
    num_flange_low, num_web, num_mid, num_side, num_half_arc, _, mesh_size_beam, _, _ = param_decoder(member)
    # Create datum planes for segmentation
    datum_plane1 = part1.DatumPlaneByPrincipalPlane(offset=d/2.-tf-r, principalPlane=XZPLANE)
    datum_plane2 = part1.DatumPlaneByPrincipalPlane(offset=-d/2.+tf+r, principalPlane=XZPLANE)
    datum_plane3 = part1.DatumPlaneByPrincipalPlane(offset=tw/2.+r, principalPlane=YZPLANE)
    datum_plane4 = part1.DatumPlaneByPrincipalPlane(offset=-tw/2.-r, principalPlane=YZPLANE)
    datum_plane5 = part1.DatumPlaneByPrincipalPlane(offset=d/2.-2./3.*tf, principalPlane=XZPLANE)
    datum_plane6 = part1.DatumPlaneByPrincipalPlane(offset=-d/2.+2./3.*tf, principalPlane=XZPLANE)
    datum_point1 = part1.DatumPointByCoordinate(coords=(tw/2., d/2.-2./3.*tf, 0.))
    datum_point2 = part1.DatumPointByCoordinate(coords=(-tw/2., d/2.-2./3.*tf, 0.))
    datum_point3 = part1.DatumPointByCoordinate(coords=(tw/2., -d/2.+2./3.*tf, 0.))
    datum_point4 = part1.DatumPointByCoordinate(coords=(-tw/2., -d/2.+2./3.*tf, 0.))
    datum_plane7 = part1.DatumPlaneByThreePoints(point1=part1.datums[datum_point1.id],
                                                 point2=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(tw/2.+(1-math.sqrt(2)/2.)*r,
                                                                                          d/2.-tf-(1-math.sqrt(2)/2.)*r, 0.)),
                                                     rule=MIDDLE),
                                                 point3=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(tw/2.+(1-math.sqrt(2)/2.)*r,
                                                                                          d/2.-tf-(1-math.sqrt(2)/2.)*r, len_solid)),
                                                     rule=MIDDLE))
    datum_plane8 = part1.DatumPlaneByThreePoints(point1=part1.datums[datum_point2.id],
                                                 point2=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(-tw/2.-(1-math.sqrt(2)/2.)*r,
                                                                                          d/2.-tf-(1-math.sqrt(2)/2.)*r, 0.)),
                                                     rule=MIDDLE),
                                                 point3=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(-tw/2.-(1-math.sqrt(2)/2.)*r,
                                                                                          d/2.-tf-(1-math.sqrt(2)/2.)*r, len_solid)),
                                                     rule=MIDDLE))
    datum_plane9 = part1.DatumPlaneByThreePoints(point1=part1.datums[datum_point3.id],
                                                 point2=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(tw/2.+(1-math.sqrt(2)/2.)*r,
                                                                                          -d/2.+tf+(1-math.sqrt(2)/2.)*r, 0.)),
                                                     rule=MIDDLE),
                                                 point3=part1.InterestingPoint(
                                                     edge=part1.edges.findAt(coordinates=(tw/2.+(1-math.sqrt(2)/2.)*r,
                                                                                          -d/2.+tf+(1-math.sqrt(2)/2.)*r, len_solid)),
                                                     rule=MIDDLE))
    datum_plane10 = part1.DatumPlaneByThreePoints(point1=part1.datums[datum_point4.id],
                                                  point2=part1.InterestingPoint(
                                                      edge=part1.edges.findAt(coordinates=(-tw/2.-(1-math.sqrt(2)/2.)*r,
                                                                                           -d/2.+tf+(1-math.sqrt(2)/2.)*r, 0.)),
                                                      rule=MIDDLE),
                                                  point3=part1.InterestingPoint(
                                                      edge=part1.edges.findAt(coordinates=(-tw/2.-(1-math.sqrt(2)/2.)*r,
                                                                                           -d/2.+tf+(1-math.sqrt(2)/2.)*r, len_solid)),
                                                      rule=MIDDLE))
    datum_plane11 = part1.DatumPlaneByPrincipalPlane(offset=d/2.-tf-(1-math.sqrt(2)/2.)*r, principalPlane=XZPLANE)
    datum_plane12 = part1.DatumPlaneByPrincipalPlane(offset=-d/2.+tf+(1-math.sqrt(2)/2.)*r, principalPlane=XZPLANE)
    datum_plane13 = part1.DatumPlaneByPrincipalPlane(offset=0, principalPlane=XZPLANE)
    # Mesh definition
    # Segment solid I-section column part, Partition 1
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane1.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane2.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane5.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane6.id])
    # Define mesh setting for solid
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((w/2., d/2.-tf/3., 0.), ), ((w/2., d/2.-tf/3., len_solid), ),
                                                    ((-w/2., d/2.-tf/3., 0.), ), ((-w/2., d/2.-tf/3., len_solid), ),
                                                    ((w/2., -d/2.+tf/3., 0.), ), ((w/2., -d/2.+tf/3., len_solid), ),
                                                    ((-w/2., -d/2.+tf/3., 0.), ), ((-w/2., -d/2.+tf/3., len_solid), )),
                           number=num_flange_up, constraint=FIXED)
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((w/2., d/2.-tf/6.*5, 0.), ), ((w/2., d/2.-tf/6.*5, len_solid), ),
                                                    ((-w/2., d/2.-tf/6.*5, 0.), ), ((-w/2., d/2.-tf/6.*5, len_solid), ),
                                                    ((w/2., -d/2.+tf/6.*5, 0.), ), ((w/2., -d/2.+tf/6.*5, len_solid), ),
                                                    ((-w/2., -d/2.+tf/6.*5, 0.), ), ((-w/2., -d/2.+tf/6.*5, len_solid), )),
                           number=num_flange_low, constraint=FREE)
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((0., d/2.-tf-r, 0.), ), ((0., d/2.-tf-r, len_solid), ),
                                                    ((0., -d/2.+tf+r, 0.), ), ((0., -d/2.+tf+r, len_solid), )),
                           number=num_web, constraint=FIXED)
    # Partition 2
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane3.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane4.id])
    # Seed the arc segments of k area
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((tw/2.+(1-math.sqrt(2)/2)*r, d/2.-tf-(1-math.sqrt(2)/2)*r, 0.), ),
                                                    ((-tw/2.-(1-math.sqrt(2)/2)*r, -d/2.+tf+(1-math.sqrt(2)/2)*r, 0.), ),
                                                    ((tw/2.+(1-math.sqrt(2)/2)*r, -d/2.+tf+(1-math.sqrt(2)/2)*r, 0.), ),
                                                    ((-tw/2.-(1-math.sqrt(2)/2)*r, d/2.-tf-(1-math.sqrt(2)/2)*r, 0.), ),
                                                    ((tw/2.+(1-math.sqrt(2)/2)*r, d/2.-tf-(1-math.sqrt(2)/2)*r, len_solid), ),
                                                    ((-tw/2.-(1-math.sqrt(2)/2)*r, -d/2.+tf+(1-math.sqrt(2)/2)*r, len_solid), ),
                                                    ((tw/2.+(1-math.sqrt(2)/2)*r, -d/2.+tf+(1-math.sqrt(2)/2)*r, len_solid), ),
                                                    ((-tw/2.-(1-math.sqrt(2)/2)*r, d/2.-tf-(1-math.sqrt(2)/2)*r, len_solid), )),
                           number=num_half_arc * 2, constraint=FIXED)
    # Partition 3
    part1.PartitionCellByDatumPlane(cells=part1.cells.findAt(coordinates=(0., d/2.-tf, len_solid/2.)),
                                    datumPlane=part1.datums[datum_plane7.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells.findAt(coordinates=(0., d/2.-tf, len_solid/2.)),
                                    datumPlane=part1.datums[datum_plane8.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells.findAt(coordinates=(0., -d/2.+tf, len_solid/2.)),
                                    datumPlane=part1.datums[datum_plane9.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells.findAt(coordinates=(0., -d/2.+tf, len_solid/2.)),
                                    datumPlane=part1.datums[datum_plane10.id])
    # Seed the middle line of k area
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((0., d/2.-tf/3.*2, 0.),), ((0., d/2.-tf/3.*2, len_solid),),
                                                    ((0., -d/2.+tf/3.*2, 0.),), ((0., -d/2.+tf/3.*2, len_solid),)),
                           number=num_mid, constraint=FREE)
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((tw/2.+r/2., d/2.-tf/3.*2, 0.),), ((tw/2.+r/2., d/2.-tf/3.*2, len_solid),),
                                                    ((tw/2.+r/2., -d/2.+tf/3.*2, 0.),), ((tw/2.+r/2., -d/2.+tf/3.*2, len_solid),),
                                                    ((-tw/2.-r/2., d/2.-tf/3.*2, 0.),), ((-tw/2.-r/2., d/2.-tf/3.*2, len_solid),),
                                                    ((-tw/2.-r/2., -d/2.+tf/3.*2, 0.),), ((-tw/2.-r/2., -d/2.+tf/3.*2, len_solid),)),
                           number=num_side, constraint=FIXED)
    # Partition 4
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane11.id])
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane12.id])
    # Seed the part
    part1.seedEdgeByNumber(edges=part1.edges.findAt(((0., d/2.-tf-(1-math.sqrt(2)/2.)*r, 0.),),
                                                    ((0., d/2.-tf-(1-math.sqrt(2)/2.)*r, len_solid),),
                                                    ((0., -d/2.+tf+(1-math.sqrt(2)/2.)*r, 0.),),
                                                    ((0., -d/2.+tf+(1-math.sqrt(2)/2.)*r, len_solid),)),
                           number=num_mid, constraint=FREE)
    # Partition 5
    part1.PartitionCellByDatumPlane(cells=part1.cells, datumPlane=part1.datums[datum_plane13.id])
    # Seed the web
    part1.seedEdgeBySize(edges=part1.edges.findAt(((tw/2., d/4.-tf/2.-r/2., 0.),), ((tw/2., d/4.-tf/2.-r/2., len_solid),),
                                                  ((-tw/2., d/4.-tf/2.-r/2., 0.),), ((-tw/2., d/4.-tf/2.-r/2., len_solid),),
                                                  ((tw/2., -d/4.+tf/2.+r/2., 0.),), ((tw/2., -d/4.+tf/2.+r/2., len_solid),),
                                                  ((-tw/2., -d/4.+tf/2.+r/2., 0.),), ((-tw/2., -d/4.+tf/2.+r/2., len_solid),)),
                         size=mesh_size_web, constraint=FREE)
    # Seed the flange
    part1.seedEdgeBySize(edges=part1.edges.findAt(((w/4.+tw/4.+r/2., d/2., 0.),), ((w/4.+tw/4.+r/2., d/2., len_solid),),
                                                  ((w/4.+tw/4.+r/2., d/2.-tf, 0.),), ((w/4.+tw/4.+r/2., d/2.-tf, len_solid),),
                                                  ((-w/4.-tw/4.-r/2., d/2., 0.),), ((-w/4.-tw/4.-r/2., d/2., len_solid),),
                                                  ((-w/4.-tw/4.-r/2., d/2.-tf, 0.),), ((-w/4.-tw/4.-r/2., d/2.-tf, len_solid),),
                                                  ((w/4.+tw/4.+r/2., -d/2., 0.),), ((w/4.+tw/4.+r/2., -d/2., len_solid),),
                                                  ((w/4.+tw/4.+r/2., -d/2.+tf, 0.),), ((w/4.+tw/4.+r/2., -d/2.+tf, len_solid),),
                                                  ((-w/4.-tw/4.-r/2., -d/2., 0.),), ((-w/4.-tw/4.-r/2., -d/2., len_solid),),
                                                  ((-w/4.-tw/4.-r/2., -d/2.+tf, 0.),), ((-w/4.-tw/4.-r/2., -d/2.+tf, len_solid),)),
                         size=mesh_size_flange, constraint=FREE)
    # Seed and mesh globally
    part1.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size_solid)
    part1.setElementType(elemTypes=(ElemType(elemCode=C3D20R, elemLibrary=STANDARD, secondOrderAccuracy=OFF,
                                             distortionControl=DEFAULT),), regions=(part1.cells,))
    part1.setMeshControls(elemShape=HEX, regions=part1.cells, technique=STRUCTURED)
    part1.generateMesh()
    # Beam segment for lateral support
    if member.type in ["Half_Beam", "Full_Beam"]:
        for offset_value in member.partition_list:
            datum_plane_b = part2.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=offset_value)
            part2.PartitionEdgeByDatumPlane(datumPlane=part2.datums[datum_plane_b.id], edges=part2.edges)
    # Define mesh setting for beam and generate mesh
    part2.setElementType(elemTypes=(ElemType(elemCode=B31OS, elemLibrary=STANDARD, secondOrderAccuracy=OFF,
                                             distortionControl=DEFAULT),), regions=(part2.edges,))
    part2.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size_beam)
    part2.generateMesh()
    # return meshed solid and beam parts
    return part1, part2


def beam_model_step_load_definer(member, set1, set2, instance3, model_name="Model-Macro"):
    """
    Define steps and boundaries (suitable for beams).
    :param member: beam or column object.
    :param set1: bottom central point.
    :param set2: top central point.
    :param instance3: middle beam instance.
    :param model_name: model name.
    """
    # Step definition
    mdb.models[model_name].StaticStep(initialInc=1e-4, maxInc=0.2, maxNumInc=int(1e5), nlgeom=ON,
                                      timePeriod=float(member.step_period), minInc=1e-8, name='Step-Cyclic', previous='Initial')
    # mdb.models[model_name].steps['Step-Cyclic'].setValues(stabilizationMagnitude=0.0002, stabilizationMethod=DAMPING_FACTOR,
    #                                                       continueDampingFactors=False, adaptiveDampingRatio=None)
    # Define user-defined load
    mdb.models[model_name].ConcentratedForce(name='Load-Amp', createStepName='Step-Cyclic', region=set2, cf3=1.0, amplitude='Amp2')
    # Define field output
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'PE', 'LE', 'U', 'SDV'))
    # Define history output
    del mdb.models[model_name].historyOutputRequests['H-Output-1']
    mdb.models[model_name].steps['Step-Cyclic'].control.setValues(allowPropagation=OFF, resetDefaultValues=OFF,
                                                                  timeIncrementation=(8.0, 10.0, 9.0, 16.0, 10.0,
                                                                                      4.0, 12.0, 10.0, 6.0, 3.0, 50.0))
    # Define history output
    mdb.models[model_name].HistoryOutputRequest(name='H-Output-BOT', createStepName='Step-Cyclic', variables=('RM1', 'RF2'),
                                                region=set1)
    mdb.models[model_name].HistoryOutputRequest(name='H-Output-TOP', createStepName='Step-Cyclic', variables=('U2', 'U3', 'UR1'),
                                                region=set2)
    # Define sensors
    mdb.models[model_name].HistoryOutputRequest(name='RMSENSOR', createStepName='Step-Cyclic', variables=('RM1',), region=set1, sensor=ON)
    mdb.models[model_name].HistoryOutputRequest(name='DPSENSOR', createStepName='Step-Cyclic', variables=('U2',), region=set2, sensor=ON)
    # Apply bottom end boundary
    mdb.models[model_name].DisplacementBC(name='BC-BOT', createStepName='Step-Cyclic', region=set1, u1=0.0, u2=0.0, u3=0.0,
                                          ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET)
    if member.type == "Half_Beam":
        # Define cyclic load
        mdb.models[model_name].DisplacementBC(name='BC-U2', createStepName='Step-Cyclic', region=set2, u1=UNSET, u2=member.mem_len, u3=UNSET,
                                              ur1=UNSET, ur2=0.0, ur3=UNSET, amplitude='Amp1')
        # Define lateral support
        if member.partition_index == 1:
            member.partition_list.append(-member.len_beam / 2)
        for i, loc in enumerate([member.mem_len / 2 - j - member.len_beam / 2 for j in member.partition_list]):
            vert = instance3.vertices.findAt(((0.0, 0.0, loc),))
            mdb.models[model_name].DisplacementBC(name='BC-L%d' % i, createStepName='Step-Cyclic', region=regionToolset.Region(vertices=vert),
                                                  u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=0.0, amplitude=UNSET)
    else:
        # Define cyclic load
        mdb.models[model_name].DisplacementBC(name='BC-U2', createStepName='Step-Cyclic', region=set2, u1=0.0, u2=member.mem_len, u3=UNSET,
                                              ur1=0.0, ur2=0.0, ur3=0.0, amplitude='Amp1')
        # Define lateral support
        for i in range(member.segments - 1):
            vert = instance3.vertices.findAt(((0.0, 0.0, (i + 1) * member.segment_length),))
            mdb.models[model_name].DisplacementBC(name='BC-L%d' % i, createStepName='Step-Cyclic', region=regionToolset.Region(vertices=vert),
                                                  u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=0.0, amplitude=UNSET)


def column_model_step_load_definer(member, set1, set2, model_name="Model-Macro"):
    """
    Define steps and boundaries (suitable for columns).
    :param member: beam or column object.
    :param set1: bottom central point.
    :param set2: top central point.
    :param model_name: model name.
    """
    # Step definition
    mdb.models[model_name].StaticStep(initialInc=1e-4, maxInc=0.2, maxNumInc=int(1e5), nlgeom=ON, timePeriod=1,
                                      minInc=1e-8, name='Step-Axial', previous='Initial')
    mdb.models[model_name].StaticStep(initialInc=1e-4, maxInc=0.2, maxNumInc=int(1e5), nlgeom=ON,
                                      timePeriod=float(member.step_period), minInc=1e-8, name='Step-Cyclic', previous='Step-Axial')
    # Define user-defined load
    mdb.models[model_name].ConcentratedForce(name='Load-Amp', createStepName='Step-Axial', region=set2, cf3=1.0, amplitude='Amp2')
    # Define field output
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'PE', 'LE', 'U', 'SDV'))
    # Define history output
    del mdb.models[model_name].historyOutputRequests['H-Output-1']
    mdb.models[model_name].steps['Step-Cyclic'].control.setValues(allowPropagation=OFF, resetDefaultValues=OFF,
                                                                  timeIncrementation=(8.0, 10.0, 9.0, 16.0, 10.0,
                                                                                      4.0, 12.0, 10.0, 6.0, 3.0, 50.0))
    # Define history output
    mdb.models[model_name].HistoryOutputRequest(name='H-Output-BOT', createStepName='Step-Axial', variables=('RM1', 'RF2'),
                                                region=set1)
    mdb.models[model_name].HistoryOutputRequest(name='H-Output-TOP', createStepName='Step-Axial', variables=('U2', 'U3', 'UR1'),
                                                region=set2)
    # Define sensors
    mdb.models[model_name].HistoryOutputRequest(name='RMSENSOR', createStepName='Step-Axial', variables=('RM1',), region=set1, sensor=ON)
    mdb.models[model_name].HistoryOutputRequest(name='DPSENSOR', createStepName='Step-Axial', variables=('U2',), region=set2, sensor=ON)
    # Apply bottom end boundary
    mdb.models[model_name].DisplacementBC(name='BC-BOT', createStepName='Step-Axial', region=set1, u1=0.0, u2=0.0, u3=0.0,
                                          ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET)
    # Apply top end boundary
    mdb.models[model_name].ConcentratedForce(name='Load-Axial', createStepName='Step-Axial', region=set2, cf3=-member.axial_load)
    mdb.models[model_name].DisplacementBC(name='BC-U2', createStepName='Step-Axial', region=set2, u1=0.0, u2=0.0, u3=UNSET,
                                          ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET)
    # Define cyclic load
    # Boundary for fixed end
    if member.boundary == "Fixed":
        mdb.models[model_name].boundaryConditions['BC-U2'].setValuesInStep(stepName='Step-Cyclic', u2=member.mem_len, amplitude='Amp1')
    # Boundary for flexible end
    if member.boundary == "Flexible":
        mdb.models[model_name].rootAssembly.engineeringFeatures.SpringDashpotToGround(name='Spring', region=set2, orientation=None, dof=4,
                                                                                      springBehavior=ON, springStiffness=member.spring)
        mdb.models[model_name].boundaryConditions['BC-U2'].setValuesInStep(stepName='Step-Cyclic', u2=member.mem_len, ur1=FREED,
                                                                           amplitude='Amp1')


def step_load_definer(member, set1, set2, instance3, model_name="Model-Macro"):
    """
    Define steps and boundaries.
    :param member: beam or column object.
    :param set1: bottom central point.
    :param set2: top central point.
    :param instance3: beam instance.
    :param model_name: model name.
    """
    if member.type in ["Half_Beam", "Full_Beam"]:
        beam_model_step_load_definer(member, set1, set2, instance3, model_name)
    elif member.type == "Column":
        column_model_step_load_definer(member, set1, set2, model_name)
    else:
        raise ValueError


def half_model_assembly_creator(member, part1, part2, model_name="Model-Macro"):
    """
    Assemble the parts and return instances and surfaces (suitable for half model cases).
    :param member: beam or column member.
    :param part1: solid element segment part.
    :param part2: beam element segment part.
    :param model_name: model name.
    :return: assembled instances amd several particular surfaces.
    """
    # Add the solid and beam instances
    Assembly = mdb.models[model_name].rootAssembly
    instance1 = Assembly.Instance(dependent=ON, name='Bot', part=part1)
    instance3 = Assembly.Instance(dependent=ON, name='Mid', part=part2)
    # Adjust the position of solid and beam instances
    Assembly.translate(instanceList=('Mid',), vector=(0., 0., member.len_beam / 2 + member.len_solid))
    Assembly.rotate(instanceList=('Mid',), axisPoint=(0., 0., member.len_beam / 2 + member.len_solid),
                    axisDirection=(0., 1., 0.), angle=90.)
    # Surfaces
    surf1 = Assembly.Surface(side1Faces=instance1.faces.findAt(((0., 0., 0.),)), name='SOLID-BOT-BOT')
    return [instance1, instance3, surf1]


def full_model_assembly_creator(member, part1, part2, model_name="Model-Macro"):
    """
    Assemble the parts and return instances and surfaces (suitable for full model cases).
    :param member: beam or column member.
    :param part1: solid element segment part.
    :param part2: beam element segment part.
    :param model_name: model name.
    :return: assembled instances amd several particular surfaces.
    """
    # Add the solid and beam instances
    Assembly = mdb.models[model_name].rootAssembly
    instance1 = Assembly.Instance(dependent=ON, name='Bot', part=part1)
    instance2 = Assembly.Instance(dependent=ON, name='Top', part=part1)
    instance3 = Assembly.Instance(dependent=ON, name='Mid', part=part2)
    # Adjust the position of solid and beam instances
    Assembly.translate(instanceList=('Top',), vector=(0., 0., member.mem_len - member.len_solid))
    Assembly.translate(instanceList=('Mid',), vector=(0., 0., member.mem_len / 2))
    Assembly.rotate(instanceList=('Mid',), axisPoint=(0., 0., member.mem_len / 2), axisDirection=(0., 1., 0.), angle=90.)
    # Surfaces
    surf1 = Assembly.Surface(side1Faces=instance1.faces.findAt(((0., 0., 0.),)), name='SOLID-BOT-BOT')
    surf2 = Assembly.Surface(side1Faces=instance2.faces.findAt(((0., 0., member.mem_len),)), name='SOLID-TOP-TOP')
    return [instance1, instance2, instance3, surf1, surf2]


def assembly_creator(member, part1, part2, model_name="Model-Macro"):
    """
    Assemble the parts and return instances and surfaces.
    :param member: beam or column member.
    :param part1: solid element segment part.
    :param part2: beam element segment part.
    :param model_name: model name.
    :return: assembled instances amd several particular surfaces.
    """
    if member.type == "Half_Beam":
        return half_model_assembly_creator(member, part1, part2, model_name)
    elif member.type in ["Full_Beam", "Column"]:
        return full_model_assembly_creator(member, part1, part2, model_name)
    else:
        raise ValueError("Member type problem!")


def residual_stress(model, xy_list, type_, a_, c_, d, b_f, t_f, t_w):
    """
    Add residual stress.
    :param model: model object.
    :param xy_list: xy_list recording the coordinates of element.
    :param type_: string indicating bottom ot top segment.
    :param a_: residual stress parameter a.
    :param c_: residual stress parameter c.
    :param d: sectional depth.
    :param b_f: sectional width.
    :param t_f: flange thickness.
    :param t_w: web thickness.
    """
    d_ = 4 * (a_ - c_) / (d - t_f) ** 2
    b_ = - (2 * t_f * b_f * a_ + t_w * (d - 2 * t_f) * c_ + t_w * d_ / 12 * (d - 2 * t_f) ** 3) / (2 * t_f * b_f ** 3 / 12)
    # Flange residual stress
    def func_flange(x):
        return a_ + b_ * x ** 2
    # Web residual stress
    def func_web(x):
        return c_ + d_ * x ** 2
    # Apply residual stress
    n = 0
    for k, func in enumerate([func_flange, func_web, func_flange]):
        for i in range(xy_list[k].shape[0]):
            for j in range(xy_list[k].shape[1]):
                xy_i = xy_list[k][i, j]
                model.Stress(name=type_ + '%d' % n, region=model.rootAssembly.sets[type_ + '%d' % n], distributionType=UNIFORM,
                             sigma11=0.0, sigma22=0.0, sigma33=func(abs(xy_i)), sigma12=0.0, sigma13=0.0, sigma23=0.0)
                n += 1


def half_model_bs_creator(member, instance1, instance3, surf1, model_name="Model-Macro"):
    """
    Create set and boundary conditions for assembled instances (suitable for half model cases).
    :param member: beam or column member.
    :param instance1: bottom solid instance.
    :param instance3: middle beam instance.
    :param surf1: bottom surface.
    :param model_name: model name.
    """
    # Read geometrical params
    d, w, tf, tw, r, mem_len, len_beam, len_solid, mesh_size_solid, mesh_size_web, mesh_size_flange, efu, \
    efl, ewm, _, _, _, _, mesh_size_beam, step_period, axial_load = param_decoder(member)
    Assembly = mdb.models[model_name].rootAssembly
    # Define key geometry, element and node sets
    for line in ss.Set_code1 * 5:
        exec(line)
    xy_list1 = ss.set_selection_bot(Assembly, nodes1, instance1, w, d, tf, tw, r, len_solid, mesh_size_solid, mesh_size_flange, efu+efl, ewm, member.scale)
    # Residual stress
    residual_stress(mdb.models[model_name], xy_list1, 'S1E', member.a_, member.c_, d, w, tf, tw)
    # Interaction definition
    # Add reference points and the coupling the surface
    mdb.models[model_name].Coupling(name='Constraint-1', controlPoint=set1, surface=surf1, influenceRadius=WHOLE_SURFACE,
                                    couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    # Define steps and load
    step_load_definer(member, set1, set2, instance3, model_name)


def full_model_bs_creator(member, instance1, instance2, instance3, surf1, surf2, model_name="Model-Macro"):
    """
    Create set and boundary conditions for assembled instances (suitable for full model cases).
    :param member: beam or column member.
    :param instance1: bottom solid instance.
    :param instance2: top solid instance.
    :param instance3: middle beam instance.
    :param surf1: bottom surface.
    :param surf2: top surface.
    :param model_name: model name.
    """
    # Read geometrical params
    d, w, tf, tw, r, mem_len, len_beam, len_solid, mesh_size_solid, mesh_size_web, mesh_size_flange, efu, \
    efl, ewm, _, _, _, _, mesh_size_beam, step_period, axial_load = param_decoder(member)
    Assembly = mdb.models[model_name].rootAssembly
    # Define key geometry, element and node sets (There seems to be a bug in set definition, so it has to be defined multiple times)
    for line in ss.Set_code2 * 5:
        exec(line)
    xy_list1 = ss.set_selection_bot(Assembly, nodes1, instance1, w, d, tf, tw, r, len_solid, mesh_size_solid, mesh_size_flange, efu+efl, ewm, member.scale)
    xy_list2 = ss.set_selection_top(Assembly, nodes2, instance2, w, d, tf, tw, r, len_solid, mesh_size_solid, mesh_size_flange, mem_len, efu+efl, ewm, member.scale)
    # Residual stress
    residual_stress(mdb.models[model_name], xy_list1, 'S1E', member.a_, member.c_, d, w, tf, tw)
    residual_stress(mdb.models[model_name], xy_list2, 'S2E', member.a_, member.c_, d, w, tf, tw)
    # Interaction definition
    # Add reference points and the coupling the surface
    mdb.models[model_name].Coupling(name='Constraint-1', controlPoint=set1, surface=surf1, influenceRadius=WHOLE_SURFACE,
                                    couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    mdb.models[model_name].Coupling(name='Constraint-2', controlPoint=set2, surface=surf2, influenceRadius=WHOLE_SURFACE,
                                    couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    # Define steps and load
    step_load_definer(member, set1, set2, instance3, model_name)


def set_boundary_creator(member, ins_surfs, model_name="Model-Macro"):
    """
    Create set and boundary conditions for assembled instances.
    :param member: beam or column member.
    :param ins_surfs: instances and surfaces.
    :param model_name: model name.
    """
    if member.type == "Half_Beam":
        half_model_bs_creator(member, ins_surfs[0], ins_surfs[1], ins_surfs[2], model_name)
    elif member.type in ["Full_Beam", "Column"]:
        full_model_bs_creator(member, ins_surfs[0], ins_surfs[1], ins_surfs[2], ins_surfs[3], ins_surfs[4], model_name)
    else:
        raise ValueError("Member type problem!")

