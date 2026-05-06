"""Load Gmsh models into pyquasar domain data structures."""

from collections.abc import Iterable

import gmsh
import numpy as np

from ._typing import DomainData


def load_mesh(
    filename: str, numPart: int = 0, refineK: int = 0
) -> Iterable[DomainData]:
    """Load and optionally partition a Gmsh mesh into pyquasar domains.

    Parameters
    ----------
    filename
        Path to a Gmsh ``.geo`` or mesh file.
    numPart
        Number of mesh partitions to request from Gmsh. ``0`` leaves the mesh
        unpartitioned.
    refineK
        Number of uniform mesh refinement steps to apply before partitioning.

    Yields
    ------
    DomainData
        Material name, boundary indices, vertices, element blocks, and boundary
        blocks for each domain entity.
    """

    def create_domain(dim: int, tag: int) -> DomainData:
        def get_material(dim: int, tag: int) -> str | None:
            physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
            assert len(physicalTags) <= 1
            return (
                gmsh.model.getPhysicalName(dim, physicalTags[0])
                if len(physicalTags)
                else None
            )

        def create_block(dim: int, tag: int):
            for elemType, _, elemNodeTag in zip(
                *gmsh.model.mesh.getElements(dim, tag), strict=True
            ):
                elemName, _, _, numNodes, *_ = gmsh.model.mesh.getElementProperties(
                    elemType
                )
                quad, weight = gmsh.model.mesh.getIntegrationPoints(elemType, "Gauss2")
                assert len(bad_indices := np.setdiff1d(elemNodeTag, nodeTags)) == 0, (
                    bad_indices,
                    elemNodeTag,
                )
                yield (
                    elemName,
                    inv_indices[elemNodeTag].reshape(-1, numNodes),
                    quad.reshape(-1, 3)[:, :dim],
                    weight,
                )

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
        for bdim, btag in gmsh.model.getBoundary([(dim, tag)], oriented=True):
            blocks = list(create_block(bdim, abs(btag)))
            assert len(blocks) > 0, (dim, tag)
            boundaries.append((get_material(bdim, abs(btag)), btag, blocks))

        return material, boundary_indices, vertices, elements, boundaries

    gmsh.initialize()
    try:
        gmsh.open(filename)
        print(f"Load model {gmsh.model.getCurrent()} ({gmsh.model.getDimension()}D)")

        gmsh.option.setNumber("Mesh.PartitionCreateTopology", 1)
        gmsh.option.setNumber("Mesh.PartitionCreatePhysicals", 1)

        if filename.endswith(".geo"):
            gmsh.model.mesh.generate(gmsh.model.getDimension())

        for _ in range(refineK):
            gmsh.model.mesh.refine()
        gmsh.model.mesh.partition(numPart)
        # gmsh.write(f'{gmsh.model.getCurrent()}.msh')

        is_interface = np.zeros(gmsh.model.mesh.getMaxNodeTag() + 1, dtype=bool)
        nodeTags, _, _ = gmsh.model.mesh.getNodes(
            gmsh.model.getDimension() - 1, -1, True, False
        )
        is_interface[nodeTags] = True
        global_indices = np.zeros(is_interface.shape, dtype=int)
        global_indices[is_interface] = np.arange(is_interface.sum())

        is_part = gmsh.model.getNumberOfPartitions() > 0
        for dim, tag in gmsh.model.getEntities(gmsh.model.getDimension()):
            if is_part:
                partitions = gmsh.model.getPartitions(dim, tag)
                if not len(partitions):
                    continue
            yield create_domain(dim, tag)
    finally:
        gmsh.clear()
        gmsh.finalize()
