import io
from typing import Generator, Optional

import numpy as np
import numpy.typing as npt
import gmsh


class MeshBlock:
  """A set of elements of the same type in a mesh."""

  def __init__(
    self,
    type: str,
    node_tags: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ) -> None:
    self._type = type
    self._node_tags = node_tags
    self._quad_points = quad_points
    self._weights = weights

  @property
  def type(self) -> str:
    """The type of the block elements."""
    return self._type

  @property
  def node_tags(self) -> npt.NDArray[np.signedinteger]:
    """The node tags of the block."""
    return self._node_tags

  @property
  def quad_points(self) -> npt.NDArray[np.floating]:
    """The quadrature points of the block."""
    return self._quad_points

  @property
  def weights(self) -> npt.NDArray[np.floating]:
    """The weights of the quadrature points."""
    return self._weights

  def __repr__(self) -> str:
    return (
      f"MeshBlock(type={repr(self.type)}, "
      f"node_tags={repr(self.node_tags)}, "
      f"quad_points={repr(self.quad_points)}, "
      f"weights={repr(self.weights)}"
    )


class MeshBoundary:
  """A class to represent a boundary in a mesh."""

  def __init__(self, type: Optional[str], tag: int, elements: list[MeshBlock]) -> None:
    self._type = type
    self._tag = tag
    self._elements = elements

  @property
  def type(self) -> Optional[str]:
    """The type of the boundary."""
    return self._type

  @property
  def tag(self) -> int:
    """The tag associated with the boundary."""
    return self._tag

  @property
  def elements(self) -> list[MeshBlock]:
    """The boundary elements."""
    return self._elements

  def __repr__(self) -> str:
    return f"MeshBoundary(type={repr(self.type)}, tag={repr(self.tag)}, elements={repr(self.elements)})"


class MeshDomain:
  """A class to represent a mesh domain."""

  def __init__(
    self,
    material: Optional[str],
    dim: int,
    tag: int,
    vertices: npt.NDArray[np.floating],
    elements: list[MeshBlock],
    boundaries: list[MeshBoundary],
    boundary_indices: npt.NDArray[np.signedinteger],
  ) -> None:
    self._material = material
    self._dim = dim
    self._tag = tag
    self._vertices = vertices
    self._elements = elements
    self._boundaries = boundaries
    self._boundary_indices = boundary_indices
    self._elements_count = sum(len(element.node_tags) for element in self.elements)

  @property
  def material(self) -> str:
    """The material of the domain."""
    return self._material

  @property
  def dim(self) -> int:
    """The dimension of the domain."""
    return self._dim

  @property
  def tag(self) -> int:
    """The tag of the domain."""
    return self._tag

  @property
  def vertices(self) -> npt.NDArray[np.floating]:
    """The vertices of the domain."""
    return self._vertices

  @property
  def elements(self) -> list[MeshBlock]:
    """The elements of the domain."""
    return self._elements

  @property
  def boundaries(self) -> list[MeshBoundary]:
    """The boundaries of the domain."""
    return self._boundaries

  @property
  def boundary_indices(self) -> npt.NDArray[np.signedinteger]:
    """Indices of the boundary nodes.

    Note
    ----
    The boundary indices are the global indices of the boundary nodes.
    """
    return self._boundary_indices

  @property
  def elements_count(self) -> int:
    """The number of elements in the domain."""
    return self._elements_count

  def __repr__(self) -> str:
    PAD = "\n\t"
    return (
      "<MeshDomain object summary"
      f"{PAD}Material: {self.material}"
      f"{PAD}Total elements number: {self.elements_count}"
      f"{PAD}{self.__elements_summary_to_string()}"
      f"{self.__boundaries_summary_to_string()}>"
    )

  def __elements_summary_to_string(self) -> str:
    result = io.StringIO("Elements:\n")
    for element in self.elements:
      result.write(f"Element type: {element.type}; Count: {element.node_tags.shape[0]}\n")
    return result.getvalue()

  def __boundaries_summary_to_string(self) -> str:
    result = io.StringIO("Boundaries:\n")
    for boundary in self.boundaries:
      btype = str(boundary.type) if boundary.type is not None else "None"
      result.write(f"\tBoundary type: {btype}; Tag: {boundary.tag}; ")
      for element in boundary.elements:
        result.write(f"Element type: {element.type}; Count: {element.node_tags.shape[0]}.\n")
    return result.getvalue()

  def _localize_numeration(self, is_interface: npt.NDArray[np.bool_], global_indices: npt.NDArray[np.int64]) -> None:
    domain_tags = []
    for element in self.elements:
      domain_tags.append(element.node_tags.flatten())
    domain_tags = np.unique(np.concatenate(domain_tags))
    inv_indices = np.empty_like(global_indices)
    mask = is_interface[domain_tags]
    local_boundary_indices = global_indices[domain_tags[mask]]
    inv_indices[domain_tags[mask]] = np.arange(local_boundary_indices.size)
    inv_indices[domain_tags[~mask]] = np.arange(local_boundary_indices.size, domain_tags.size)
    local_vertices = np.empty((domain_tags.size, self.dim))
    local_vertices[inv_indices[domain_tags]] = self.vertices[domain_tags]
    self._vertices = local_vertices
    self._boundary_indices = local_boundary_indices
    for block in self.elements:
      block._node_tags = inv_indices[block.node_tags]
    for boundary in self.boundaries:
      for block in boundary.elements:
        block._node_tags = inv_indices[block.node_tags]


class Mesh:
  """A class to represent a mesh."""

  def __init__(self, domains: list[MeshDomain]):
    self._domains = domains
    self._current = 0
    self._numeration = "global"

  @property
  def domains(self) -> list[MeshDomain]:
    """The domains of the mesh."""
    return self._domains

  def __iter__(self):
    return self

  def __next__(self) -> MeshDomain:
    if self._current >= len(self.domains):
      raise StopIteration
    domain = self.domains[self._current]
    self._current += 1
    return domain

  def __repr__(self) -> str:
    PAD = "\n\t"
    return f"<Mesh object summary {PAD}Numeration: {self.numeration}{PAD}Domains: {repr(self.domains)}>"

  @property
  def numeration(self) -> str:
    """The numeration of the mesh.

    Note
    ----
    Local numeration means that the indices of the elements are local to the domain. Needs to be local for the FETI problem.
    """
    return self._numeration

  @classmethod
  def load(cls, file: str, refine_k: int = 0, num_part: int = 0):
    """Load a mesh from a file and generate domains.

    Parameters
    ----------
    file : str
      The name of the file from which to load the mesh.
    refine_k : int, optional
      The number of times to refine the mesh, by default 0.
    num_part : int, optional
      The number of partitions to create in the mesh, by default 0.

    Returns
    -------
    Mesh
      A class containing the domains of the mesh.
    """

    def get_material(dim: int, tag: int) -> str | None:
      physical_tags = gmsh.model.get_physical_groups_for_entity(dim, tag)
      assert len(physical_tags) <= 1
      return gmsh.model.get_physical_name(dim, physical_tags[0]) if len(physical_tags) else None

    def generate_block(dim: int, tag: int) -> Generator[MeshBlock, tuple[int, int], None]:
      for element_type, _, element_node_tags in zip(*gmsh.model.mesh.get_elements(dim, tag)):
        element_name, _, _, num_nodes, *_ = gmsh.model.mesh.get_element_properties(element_type)
        element_node_tags = np.asarray(element_node_tags, dtype=np.int64)
        nodes_tags = (element_node_tags - 1).reshape(-1, num_nodes)
        quad_points, weights = gmsh.model.mesh.get_integration_points(element_type, "Gauss4")
        yield MeshBlock(element_name, nodes_tags, np.asarray(quad_points).reshape(-1, 3)[:, :dim], np.asarray(weights))

    def generate_domain(dim: int, tag: int) -> MeshDomain:
      material = get_material(dim, tag)
      node_tags, _, _ = gmsh.model.mesh.get_nodes(dim, tag, True, False)
      node_tags = np.asarray(node_tags, dtype=np.int64)
      vertices = np.empty((node_tags.size, dim))
      boundary_node_tags, _, _ = gmsh.model.mesh.get_nodes(gmsh.model.get_dimension() - 1, -1, True, False)
      boundary_node_tags = np.asarray(boundary_node_tags, dtype=np.int64)
      tags, coords, _ = gmsh.model.mesh.get_nodes(-1, -1, False, False)
      tags = np.asarray(tags, dtype=np.int64)
      coords = np.asarray(coords, dtype=np.float64)
      vertices = coords.reshape(-1, 3)[np.argsort(tags), :dim]
      boundary_indices = np.unique(boundary_node_tags - 1)

      elements = list(generate_block(dim, tag))
      boundaries = []
      for bdim, btag in gmsh.model.get_boundary([(dim, tag)]):
        blocks = list(generate_block(bdim, abs(btag)))
        assert len(blocks) > 0, (dim, tag)
        boundaries.append(MeshBoundary(get_material(bdim, abs(btag)), btag, blocks))

      return MeshDomain(material, dim, tag, vertices, elements, boundaries, boundary_indices)

    domains = []
    gmsh.initialize()
    gmsh.option.set_number("General.Terminal", 0)
    try:
      gmsh.open(file)

      gmsh.option.set_number("Mesh.PartitionCreateTopology", 1)
      gmsh.option.set_number("Mesh.PartitionCreatePhysicals", 1)

      if file.endswith(".geo"):
        gmsh.model.mesh.generate(gmsh.model.get_dimension())

      for _ in range(refine_k):
        gmsh.model.mesh.refine()
      gmsh.model.mesh.partition(num_part)

      is_part = gmsh.model.get_number_of_partitions() > 0
      for dim, tag in gmsh.model.get_entities(gmsh.model.get_dimension()):
        if is_part:
          partitions = gmsh.model.get_partitions(dim, tag)
          if not len(partitions):
            continue
        domains.append(generate_domain(dim, tag))

    finally:
      gmsh.clear()
      gmsh.finalize()

    return cls(domains)

  def localize_numeration(self) -> None:
    """Localize the mesh nodes numeration."""
    all_tags = []
    for domain in self.domains:
      for block in domain.elements:
        all_tags.append(block.node_tags.flatten())
    max_tag = np.max(np.unique(np.concatenate(all_tags)))
    is_interface = np.zeros(max_tag + 1, dtype=bool)
    is_interface[self.domains[0].boundary_indices] = True
    global_indices = np.zeros(is_interface.shape, dtype=np.signedinteger)
    global_indices[is_interface] = np.arange(is_interface.sum())
    for domain in self.domains:
      domain._localize_numeration(is_interface, global_indices)
    self._numeration = "local"
