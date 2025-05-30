import click
from pathlib import Path
from openfe import LigandNetwork


def get_ligandnetwork_from_graphml(graphml_path):
    """
    Get a ligand network from a path to a graphml.
    """
    with open(graphml_path) as f:
        graphml = f.read()
    return LigandNetwork.from_graphml(graphml)


def get_largest_uniques(network):
    """
    Get the edge with the highest number of unique atoms
    being transformed.
    """
    sized_edges = []
    for edge in network.edges:
        uniquesA = len(list(edge.componentA_unique))
        uniquesB = len(list(edge.componentB_unique))
        sized_edges.append((uniquesA+uniquesB, edge))

    return sorted(sized_edges, key=lambda x: x[0])[-1][1]


def process_system(graphml, name):
    network = get_ligandnetwork_from_graphml(graphml)
    edge = get_largest_uniques(network)
    edge.to_json(f"{name}_edge.json")


@click.command
@click.option(
    '--graphml',
    default=Path('./ligand_network_elf10.graphml'),
    type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
    required=True,

)
@click.option(
    '--name',
    default='name',
    type=str,
    required=True
)
def run(graphml, name):
    process_system(graphml, name)


if __name__ == "__main__":
    run()
