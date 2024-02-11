import numpy as np
import gmsh

def load_mesh(filename, numPart=0, refineK=0):
  def create_domain(dim, tag):
    def get_material(dim, tag):
      physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
      assert len(physicalTags) <= 1
      return gmsh.model.getPhysicalName(dim, physicalTags[0]) if len(physicalTags) else None

    def create_block(dim, tag):
      for elemType, _, elemNodeTag in zip(*gmsh.model.mesh.getElements(dim, tag)):
        elemName, _, _, numNodes, *_ = gmsh.model.mesh.getElementProperties(elemType)
        quad, weight = gmsh.model.mesh.getIntegrationPoints(elemType, 'Gauss2')
        assert len(bad_indices := np.setdiff1d(elemNodeTag, nodeTags)) == 0, (bad_indices, elemNodeTag)
        yield elemName, inv_indices[elemNodeTag].reshape(-1, numNodes), quad.reshape(-1, 3)[:, :dim], weight

    material = get_material(dim, tag)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim, tag, True, False)
    mask = is_interface[nodeTags]
    boundary_indices = global_indices[nodeTags[mask]]

    inv_indices = np.empty_like(global_indices)
    inv_indices[nodeTags[mask]] = np.arange(boundary_indices.size)
    inv_indices[nodeTags[~mask]] = np.arange(boundary_indices.size, nodeTags.size)

    vertices = np.empty((nodeTags.size, dim))
    vertices[inv_indices[nodeTags]] = nodeCoords.reshape(-1, 3)[:, :dim]

    elements = list(create_block(dim, tag))

    boundaries = []
    for bdim, btag in gmsh.model.getBoundary([(dim, tag)]):
      blocks = list(create_block(bdim, abs(btag)))
      #print(bdim, btag, gmsh.model.getType(bdim, abs(btag)), gmsh.model.getParent(bdim, abs(btag)),
      #      gmsh.model.getPartitions(bdim, abs(btag)), gmsh.model.getPhysicalGroupsForEntity(bdim, abs(btag)), len(blocks))
      assert len(blocks) > 0, (dim, tag)
      boundaries.append((get_material(bdim, abs(btag)), btag, blocks))

    return material, boundary_indices, vertices, elements, boundaries

  gmsh.initialize()
  try:
      gmsh.open(filename)
      print(f'Load model {gmsh.model.getCurrent()} ({gmsh.model.getDimension()}D)')

      gmsh.option.setNumber("Mesh.PartitionCreateTopology", 1)
      gmsh.option.setNumber("Mesh.PartitionCreatePhysicals", 1)

      if filename.endswith('.geo'):
        gmsh.model.mesh.generate(gmsh.model.getDimension())

      for i in range(refineK):
        gmsh.model.mesh.refine()
      gmsh.model.mesh.partition(numPart)
      #gmsh.write(f'{gmsh.model.getCurrent()}.msh')

      is_interface = np.zeros(gmsh.model.mesh.getMaxNodeTag() + 1, dtype=bool)
      nodeTags, _, _ = gmsh.model.mesh.getNodes(gmsh.model.getDimension() - 1, -1, True, False)
      is_interface[nodeTags] = True
      global_indices = np.zeros(is_interface.shape, dtype=int)
      global_indices[is_interface] = np.arange(is_interface.sum())

      is_part = gmsh.model.getNumberOfPartitions() > 0
      for dim, tag in gmsh.model.getEntities(gmsh.model.getDimension()):
        if is_part:
          partitions = gmsh.model.getPartitions(dim, tag)
          if not len(partitions):
            continue
        #print(dim, tag, gmsh.model.getType(dim, tag), gmsh.model.getPartitions(dim, tag),
        #      gmsh.model.getAdjacencies(dim, tag), gmsh.model.getPhysicalGroupsForEntity(dim, tag))
        yield create_domain(dim, tag)
  finally:
    gmsh.clear()
    gmsh.finalize()
