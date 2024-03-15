"""
Contains all functions related to visualizing neural networks.
"""

# External Visibility
__all__ = ["draw_random_network", "get_model_info"]

# Module imports
import logging
from random import uniform
from matplotlib.pyplot import axis, tight_layout, savefig
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels
from networkx import draw_networkx_nodes, spring_layout

# Set Logger and Logging Level
# logger = logging.getLogger('Spectare')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='spectare.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Function Definitions
def calculate_color(parameter: float = 0.0, bounds: tuple = (-1, 1)) -> str:
    """
    Generates a hex color code based on the
    given parameter and its intended range.
    """
    # Extract min and max values from range
    min_val, max_val = bounds[0], bounds[-1]

    # Generate random paremeter
    parameter = uniform(min_val, max_val)

    # Normalize the value to a range of [0, 1]
    normalized_value = (parameter + 1) / 2

    # Calculate red, green, and blue components
    red = round(255 * (1 - normalized_value))
    green = round(255 * normalized_value)
    blue = round(128 * (1 - abs(parameter)))  # Higher blue when value is close to 0

    # Convert to hexadecimal and ensure it's 2 characters long
    hex_red = format(red, '02x')
    hex_green = format(green, '02x')
    hex_blue = format(blue, '02x')

    # Construct full hex color
    hex_color = f'#{hex_red}{hex_green}{hex_blue}'

    return hex_color


def calculate_cb_color(parameter: float = 0.0, bounds: tuple = (-1, 1)) -> str:
    """
    Generates a colorblind-friendly hex color code based
    on the given parameter and its intended range.
    """
    # Extract min and max values from range
    min_val, max_val = bounds[0], bounds[-1]

    # Generate random paremeter
    parameter = uniform(min_val, max_val)

    # Normalize the value to a range of [0, 1]
    normalized_value = (parameter + 1) / 2

    # Calculate red, green, and blue components
    red = round(255 * (1 - normalized_value))
    green = round(128 * (1 - abs(parameter)))  # Higher blue when value is close to 0
    blue = round(255 * normalized_value)

    # Convert to hexadecimal and ensure it's 2 characters long
    hex_red = format(red, '02x')
    hex_blue = format(blue, '02x')
    hex_green = format(green, '02x')

    # Construct full hex color
    hex_color = f'#{hex_red}{hex_green}{hex_blue}'

    return hex_color


def draw_random_network(num_layers: int, num_nodes: list[int], filename: str = "Network Graph.png", colorblind: bool = False) -> None:
    """
    Draws a directed graph of a network
    with the given parameters and export
    the resulting graph to an image file.

    Args:
        num_layers (int): The number of layers in the network.
        num_nodes (list[int]): The number of nodes in each layer.

    Returns:
        None
    """
    # Check if the number of layers and nodes match
    assert len(num_nodes) == num_layers, f"Number of layers do not match: {num_layers} and {len(num_nodes)}."

    # Create a directed graph
    g = DiGraph()
    pos = spring_layout(g)

    # Calculate maximum nodes
    max_nodes = max(num_nodes)  # Maximum number of nodes in a single layer

    # Add nodes and edges
    for i in range(num_layers):
        for j in range(num_nodes[i]):
            if num_nodes[i] == 1:
                node_name = f"a[{i}]"
            else:
                node_name = f"a{j+1}[{i}]"
            logger.debug(f"Adding node: {node_name}")
            
            # Center nodes vertically
            y_pos = -(j - (num_nodes[i] - 1) / 2.0 + (max_nodes - 1) / 2.0)
            pos[node_name] = (i, y_pos)

            # Add edges from previous layer to current node
            if i > 0:
                for k in range(num_nodes[i-1]):
                    if num_nodes[i-1] == 1:
                        g.add_edge(f"a[{i-1}]", node_name)
                    else:
                        g.add_edge(f"a{k+1}[{i-1}]", node_name)
                    logger.debug(f"Adding edge: a{k+1}[{i-1}] -> {node_name}")

    # Draw the graph
    for i, (nodes) in enumerate(g.nodes()):
        node_color = calculate_color() if not colorblind else calculate_cb_color()
        draw_networkx_nodes(
            g, pos, nodelist=[nodes],
            node_size=2000, node_color=node_color)
        logger.debug(f"Drawing node: {nodes}")
    draw_networkx_labels(g, pos, font_size=8, font_color="black")
    for i, (from_node, to_node) in enumerate(g.edges()):
        node_color = calculate_color() if not colorblind else calculate_cb_color()
        draw_networkx_edges(
            g, pos, edgelist=[(from_node, to_node)],
            edge_color=node_color)
        logger.debug(f"Drawing edge: {from_node} -> {to_node}")

    # Set the axis and layout
    axis("off")
    tight_layout()

    # Save the graph to an image file
    savefig(filename, dpi=300)
    logger.info(f"Neural Network Graph saved to '{filename}'.")


def get_model_info(model) -> dict:
    """
    Extracts information about the model's
    architecture and parameters.

    Args:
        model: The PyTorch model to extract information from.

    Returns:
        tuple: A tuple containing the model's architecture and parameters.
    """
    # Get Input & Output Sizes
    input_size = model[0].in_features
    output_size = model[-1].out_features

    # Get Hidden Layer Neuron Counts
    hidden_sizes = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            hidden_sizes.append(param.shape[0])

    # Return Model Information
    return {
        "num_layers": len(hidden_sizes) + 1,
        "num_nodes": [input_size] + hidden_sizes,
    }