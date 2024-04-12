"""
Contains all functions related to visualizing neural networks.
"""

# External Visibility
__all__ = ["calculate_node_size", "detect_framework", "draw_network", "draw_tf_network", "draw_random_network", "get_model_info", "get_tf_model_info", "get_model_params", "get_tf_model_params"]

# Module imports
import logging
from math import log
from random import uniform
from time import time
t1 = time()
from typing import Dict, Tuple
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, to_hex, TwoSlopeNorm
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, Normalize, tight_layout, savefig
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels
from networkx import draw_networkx_nodes, spring_layout, kamada_kawai_layout, random_layout
import numpy as np
t2 = time()
# print(f"Module Import Time: {(t2 - t1):.4f}s")

# Set Logger and Logging Level
# logger = logging.getLogger('Spectare')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='spectare.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Function Definitions
def detect_framework() -> str:
    """
    Detects the installed deep learning framework.
    """
    torch_installed = tf_installed = False

    # Detect PyTorch
    try:
        import torch
        torch_installed = True
    except ImportError:
        pass

    # Detect TensorFlow
    try:
        import tensorflow
        tf_installed = True
    except ImportError:
        pass

    # Return the detected framework
    if torch_installed and tf_installed:
        return "both"
    elif torch_installed:
        return "torch"
    elif tf_installed:
        return "tf"
    else:
        return "none"


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


def decide_color(parameter: float = 0.0, colorblind: bool = False) -> str:
    """
    Decides which color to output
    depending on polarity of parameter.
    """
    # Check if the colorblind flag is set
    if parameter < 0:
        return "#ff0000"
    elif parameter > 0:
        return "#00ff00" if not colorblind else "#0000ff"
    else:
        return "#dddddd"
    

def float_to_red_green_color(value):
    # Ensure value is within range [-1, 1]
    value = max(-1, min(1, value))
    
    # Map the value from range [-1, 1] to range [0, 1]
    normalized_value = (value + 1) / 2
    
    # Interpolate between red and green based on the normalized value
    r = int((1 - normalized_value) * 255)
    g = int(normalized_value * 255)
    
    # Return the RGB color as a hex value
    if value == 0.0:
        return "#888888"
    return "#{:02x}{:02x}{:02x}".format(r, g, 0)


def float_to_red_blue_color(value):
    # Ensure value is within range [-1, 1]
    value = max(-1, min(1, value))
    
    # Map the value from range [-1, 1] to range [0, 1]
    normalized_value = (value + 1) / 2
    
    # Interpolate between red and green based on the normalized value
    r = int((1 - normalized_value) * 255)
    b = int(normalized_value * 255)
    
    # Return the RGB color as a hex value
    if value == 0.0:
        return "#888888"
    return "#{:02x}{:02x}{:02x}".format(r, 0, b)


def calculate_color_with_twoslope(value: float, norm: Normalize, colorblind: bool, white_neutral: bool, bounds: tuple = (-1, 1)) -> str:
    """
    Calculates a color based on the given
    value in a two-slope colormap.

    Args:
        value (float): The value to calculate the color for.
        norm (Normalize): The normalization object to use.
        bounds (tuple): The bounds of the colormap.

    Returns:
        str: The calculated color.
    """
    # Handle input neurons
    # if value == 0.0:
    #     return "#DDDDDD" if colorblind else "#AAAAAA"
    
    # Extract minimum and maximum values from bounds
    min_val, max_val = bounds[0], bounds[-1]

    # Create two-slope colormap and scalar mappable object
    norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    sm = ScalarMappable(cmap=get_cmap(colorblind, white_neutral), norm=norm)

    # Convert value to color
    rgba = sm.to_rgba(value)

    return to_hex(rgba)


def get_cmap(colorblind: bool, white_neutral: bool) -> LinearSegmentedColormap:
    """
    Returns a colormap based on the given
    colorblind and white neutral flags.

    Args:
        colorblind (bool): Whether to use colorblind-friendly colors.
        white_neutral (bool): Whether to use a one-slope or two-slope colormap.

    Returns:
        LinearSegmentedColormap: The generated colormap.

    Example:
        ```python
        cmap = get_cmap(colorblind=True, white_neutral=False)
        ```
    """
    if colorblind:
        if white_neutral:
            cmap = LinearSegmentedColormap.from_list("red_white_blue", ["#ff0000", "#dddddd", "#0000ff"])
        else:
            cmap = LinearSegmentedColormap.from_list("red_blue", ["#ff0000", "#0000ff"])
    else:
        if white_neutral:
            cmap = LinearSegmentedColormap.from_list("red_white_green", ["#ff0000", "#dddddd", "#00ff00"])
        else:
            cmap = LinearSegmentedColormap.from_list("red_green", ["#ff0000", "#00ff00"])
    return cmap


def calculate_node_size(max_num_nodes, base_node_size: int, scale_factor: int) -> int:
    """
    Calculates the size of the nodes based on the
    base node size, scale factor, and maximum number
    of nodes in a single layer.
    
    Args:
        max_num_nodes (int): The maximum number of nodes in a single layer.
        base_node_size (int): The base size of the nodes.
        scale_factor (int): The scaling factor for the node size.

    Returns:
        int: The calculated node size.
    
    Example:
        ```python3
        node_size = calculate_node_size(2000, 0.5, 5)
        ```
    """
    if max_num_nodes > 6:
        if max_num_nodes >= 10:
            logger.warning("Maximum number of nodes >= 10. Nodes may appear too small.")
        if max_num_nodes >= 18:
            logger.warning("Visualising networks with more than 18 nodes may be difficult.")
        node_size = base_node_size - scale_factor * log(max_num_nodes)
        return node_size
    return base_node_size


def update_param_bound(current_param: float, new_param: float, direction: str) -> float:
    """
    Compares a parameter value with the current min
    or max and returns the updated parameter value.

    Args:
        current_param (float): The current min or max parameter value.
        new_param (float): The new parameter value.
        direction (str): The direction of comparison (min or max).

    Returns:
        float: The updated min or max parameter value.

    Example:
        ```python
        max_param = update_param_bound(0.0, 0.5, "max")
        ```
    """
    if direction == "min":
        return min(current_param, new_param)
    elif direction == "max":
        return max(current_param, new_param)
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'min' or 'max'.")


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

    Example:
        ```python3
        draw_random_network(4, [3, 4, 5, 2], "Network Graph.png", False)
        ```
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
            logger.info(f"Adding node: {node_name}")
            
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
                    logger.info(f"Adding edge: a{k+1}[{i-1}] -> {node_name}")

    # Draw the graph
    for i, (nodes) in enumerate(g.nodes()):
        node_color = calculate_color() if not colorblind else calculate_cb_color()
        draw_networkx_nodes(
            g, pos, nodelist=[nodes],
            node_size=2000, node_color=node_color)
        logger.info(f"Drawing node: {nodes}")
    draw_networkx_labels(g, pos, font_size=8, font_color="black")
    for i, (from_node, to_node) in enumerate(g.edges()):
        node_color = calculate_color() if not colorblind else calculate_cb_color()
        draw_networkx_edges(
            g, pos, edgelist=[(from_node, to_node)],
            edge_color=node_color)
        logger.info(f"Drawing edge: {from_node} -> {to_node}")

    # Set the axis and layout
    axis("off")
    tight_layout()

    # Save the graph to an image file
    savefig(filename, dpi=300)
    logger.info(f"Neural Network Graph saved to '{filename}'.")


def draw_network(num_layers: int, num_nodes: list[int], model, filename: str = "Network Graph.png", node_base_size: int = 2000, node_size_scaling_factor: int = 640, colorblind: bool = False, draw_labels: bool = True, draw_legend: bool = True, white_neutral: bool = True) -> None:
    """
    Draws a directed graph of a model
    with the given parameters and exports
    the resulting graph to an image file.

    Args:
        num_layers (int): The number of layers in the network.
        num_nodes (list[int]): The number of nodes in each layer.
        model: The PyTorch model to extract information from.
        filename (str): The name of the image file to save the graph to.
        node_base_size (int): The base size of the nodes.
        node_size_scaling_factor (int): The scaling factor for the node size.
        colorblind (bool): Whether to use colorblind-friendly colors.
        draw_labels (bool): Whether to draw node labels.
        draw_legend (bool): Whether to draw a color legend.
        white_neutral (bool): Whether to use a one-slope or two-slope colormap.

    Returns:
        None

    Example:
        ```python
        spectare.draw_network(
            num_layers = 4,
            num_nodes = [3, 4, 5, 2],
            model = model,
            filename = "Network Graph.png",
            node_base_size = 2000,
            node_size_Scaling_factor = 50,
            colorblind = False,
            draw_labels = True,
            draw_legend = True,
            white_neutral = True)
        ```
    """
    # Check if the number of layers and nodes match
    assert len(num_nodes) == num_layers, f"Number of layers do not match: {num_layers} and {len(num_nodes)}."

    # Get model weights and biases
    model_weights = get_model_params(model, "weight")
    model_biases = get_model_params(model, "bias")

    # Create a directed graph
    g = DiGraph()
    pos = spring_layout(g)

    # Calculate maximum nodes
    max_nodes = max(num_nodes)  # Maximum number of nodes in a single layer

    # Initialize min and max param values
    min_param = float('inf')
    max_param = float('-inf')

    # Create node names and organize them into layers
    node_layers: list[Dict] = []
    for layer_index in range(len(num_nodes)):
        nodes: Dict[str, float] = {}
        for neuron_index in range(num_nodes[layer_index]):
            if num_nodes[layer_index] == 1: # if only one node in layer
                node_name = f"a[{layer_index}]"
            else: # if multiple nodes in layer
                node_name = f"a{neuron_index+1}[{layer_index}]"
            # Calculate position for node
            y_pos = -(neuron_index - (num_nodes[layer_index] - 1) / 2.0 + (max_nodes - 1) / 2.0)
            pos[node_name] = (layer_index, y_pos)
            # Add node to layer
            if layer_index == 0: # if input layer
                nodes[node_name] = 0.0
            else:
                param_value = list(model_biases.values())[layer_index-1][neuron_index-1].item()
                nodes[node_name] = param_value
                # Update min and max param values
                min_param = update_param_bound(min_param, param_value, "min")
                max_param = update_param_bound(max_param, param_value, "max")
        node_layers.append(nodes)
        logger.info(f"Adding layer {layer_index}: {nodes}")
    # print(f"Node layers:\n {node_layers}")

    # Create edges between nodes
    edge_layers: list[Dict] = []
    for layer_index, weights_by_layer in zip(range(1, len(node_layers)), list(model_weights.values())): # skip the first layer
        connections: Dict[Tuple, float] = {}
        for end_node, weights_by_end_node in zip(node_layers[layer_index], weights_by_layer):
            for start_node, start_node_weight in zip(node_layers[layer_index-1], weights_by_end_node):
                g.add_edge(start_node, end_node)
                param_value = start_node_weight.item()
                # Update min and max param values
                min_param = update_param_bound(min_param, param_value, "min")
                max_param = update_param_bound(max_param, param_value, "max")
                connections[(start_node, end_node)] = param_value
                logger.info(f"Adding layer {layer_index-1}-{layer_index} edge: {start_node} -> {end_node} ({start_node_weight})")
        edge_layers.append(connections)
    # print(f"Edge layers:\n {edge_layers}")
        
    # Collapse node and edge dictionaries into single dictionaries
    flattened_nodes = {node: bias for layer in node_layers for node, bias in layer.items()}
    flattened_edges = {edge: weight for layer in edge_layers for edge, weight in layer.items()}

    # Calculate node size
    node_size = calculate_node_size(max_nodes, node_base_size, node_size_scaling_factor)

    # Create new figure and axes
    fig, ax = plt.subplots()

    print(f"Colorblind Mode: {colorblind}, White Neutral: {white_neutral}")

    # Get colormap
    cmap = get_cmap(colorblind=colorblind, white_neutral=white_neutral)

    # Draw graph nodes
    for node in g.nodes():
        if colorblind:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                node_color = calculate_color_with_twoslope(flattened_nodes[node], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                node_color = float_to_red_blue_color(flattened_nodes[node])
        else:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                node_color = calculate_color_with_twoslope(flattened_nodes[node], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                node_color = float_to_red_green_color(flattened_nodes[node])
        draw_networkx_nodes(g, pos, nodelist=[node], node_size=node_size, node_color=node_color)
        logger.info(f"Drawing node: {node} ({node_color})")
    
    # Draw node labels
    draw_networkx_labels(g, pos, font_size=8, font_color="black") if draw_labels else None

    # Draw graph edges
    for edge in g.edges():
        if colorblind:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                edge_color = calculate_color_with_twoslope(flattened_edges[edge], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                edge_color = float_to_red_blue_color(flattened_edges[edge])
        else:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                edge_color = calculate_color_with_twoslope(flattened_edges[edge], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                edge_color = float_to_red_green_color(flattened_edges[edge]) if not colorblind else float_to_red_blue_color(flattened_edges[edge])
        draw_networkx_edges(g, pos, edgelist=[edge], edge_color=edge_color, arrows=False)
        logger.info(f"Drawing edge: {edge} ({edge_color})")
        
    # Draw legend if requested
    if draw_legend:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add a colorbar
        plt.colorbar(sm, ax=ax, label="Parameter Value")

    # Set the axis and layout
    axis("off")
    tight_layout()

    # Add edge padding
    plt.margins(0.1)

    # Save the graph to an image file
    savefig(filename, dpi=300)
    logger.info(f"Neural Network Graph saved to '{filename}'.")


def draw_tf_network(num_layers: int, num_nodes: list[int], model, filename: str = "Network Graph.png", node_base_size: int = 2000, node_size_scaling_factor: int = 640, colorblind: bool = False, draw_labels: bool = True, draw_legend: bool = True, white_neutral: bool = True) -> None:
    """
    Draws a directed graph of a TensorFlow
    model with the given parameters and exports
    the resulting graph to an image file.

    Args:
        `num_layers` (int): The number of layers in the network.
        `num_nodes` (list[int]): The number of nodes in each layer.
        `model`: The TensorFlow model to extract information from.
        `filename` (str): The name of the image file to save the graph to.
        `node_base_size` (int): The base size of the nodes.
        `node_size_scaling_factor` (int): The scaling factor for the node size.
        `colorblind` (bool): Whether to use colorblind-friendly colors.
        `draw_labels` (bool): Whether to draw node labels.
        `draw_legend` (bool): Whether to draw a color legend.
        `white_neutral` (bool): Whether to use a one-slope or two-slope colormap.

    Returns:
        None

    Example:
        ```python
        spectare.draw_tf_network(
            num_layers = 4,
            num_nodes = [3, 4, 5, 2],
            model = model,
            filename = "Network Graph.png",
            node_base_size = 2000,
            node_size_Scaling_factor = 50,
            colorblind = False,
            draw_labels = True,
            draw_legend = True,
            white_neutral = True)
        ```
    """
    # Check if the number of layers and nodes match
    assert len(num_nodes) == num_layers, f"Number of layers do not match: {num_layers} and {len(num_nodes)}."

    # Get model weights and biases
    model_weights = get_tf_model_params(model, "weight")
    model_biases = get_tf_model_params(model, "bias")

    # Create a directed grapg
    g = DiGraph()
    pos = spring_layout(g)

    # Calculate maximum nodes
    max_nodes = max(num_nodes)  # Maximum number of nodes in a single layer

    # Initialize min and max param values
    min_param = float('inf')
    max_param = float('-inf')

    logger.info(f"Drawing a TensorFlow graph with a {type(model)} model.")

    # Create node names and organize them into layers
    node_layers: list[Dict] = []
    for layer_index in range(len(num_nodes)):
        nodes: Dict[str, float] = {}
        for neuron_index in range(num_nodes[layer_index]):
            if num_nodes[layer_index] == 1:
                node_name = f"a[{layer_index}]"
            else:
                node_name = f"a{neuron_index+1}[{layer_index}]"
            # Calculate position for node
            y_pos = -(neuron_index - (num_nodes[layer_index] - 1) / 2.0 + (max_nodes - 1) / 2.0)
            pos[node_name] = (layer_index, y_pos)
            # Add node to layer
            if layer_index == 0:
                nodes[node_name] = 0.0
            else:
                param_value = list(model_biases.values())[layer_index-1][neuron_index].item()
                nodes[node_name] = param_value
                # Update min and max param values
                min_param = update_param_bound(min_param, param_value, "min")
                max_param = update_param_bound(max_param, param_value, "max")
        node_layers.append(nodes)
        logger.info(f"Adding layer {layer_index}: {nodes}")
    # print(f"Node layers:\n {node_layers}")

    # Transpose the weights to match the node layers
    model_weights = {k: v.T for k, v in model_weights.items()}

    # Create edges between nodes
    edge_layers: list[Dict] = []
    for layer_index, weights_by_layer in zip(range(1, len(node_layers)), list(model_weights.values())): # skip the first layer
        connections: Dict[Tuple, float] = {}
        for end_node, weights_by_end_node in zip(node_layers[layer_index], weights_by_layer):
            for start_node, start_node_weight in zip(node_layers[layer_index-1], weights_by_end_node):
                g.add_edge(start_node, end_node)
                param_value = start_node_weight.item()
                # Update min and max param values
                min_param = update_param_bound(min_param, param_value, "min")
                max_param = update_param_bound(max_param, param_value, "max")
                connections[(start_node, end_node)] = param_value
                logger.info(f"Adding layer {layer_index-1}-{layer_index} edge: {start_node} -> {end_node} ({start_node_weight})")
        edge_layers.append(connections)
    # print(f"Edge layers:\n {edge_layers}")

    # Collapse node and edge dictionaries into single dictionaries
    flattened_nodes = {node: bias for layer in node_layers for node, bias in layer.items()}
    flattened_edges = {edge: weight for layer in edge_layers for edge, weight in layer.items()}

    # Calculate node size
    node_size = calculate_node_size(max_nodes, node_base_size, node_size_scaling_factor)

    # Create new figure and axes
    fig, ax = plt.subplots()

    print(f"Colorblind Mode: {colorblind}, White Neutral: {white_neutral}")

    # Get colormap
    cmap = get_cmap(colorblind=colorblind, white_neutral=white_neutral)

    # Draw graph nodes
    for node in g.nodes():
        if colorblind:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                node_color = calculate_color_with_twoslope(flattened_nodes[node], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                node_color = float_to_red_blue_color(flattened_nodes[node])
        else:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                node_color = calculate_color_with_twoslope(flattened_nodes[node], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                node_color = float_to_red_green_color(flattened_nodes[node])
        draw_networkx_nodes(g, pos, nodelist=[node], node_size=node_size, node_color=node_color)
        logger.info(f"Drawing node: {node} ({node_color})")

    # Draw node labels
    draw_networkx_labels(g, pos, font_size=8, font_color="black") if draw_labels else None

    # Draw graph edges
    for edge in g.edges():
        if colorblind:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                edge_color = calculate_color_with_twoslope(flattened_edges[edge], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                edge_color = float_to_red_blue_color(flattened_edges[edge])
        else:
            if white_neutral:
                norm = TwoSlopeNorm(vmin=min_param, vcenter=0, vmax=max_param)
                edge_color = calculate_color_with_twoslope(flattened_edges[edge], norm, colorblind, white_neutral)
            else:
                norm = Normalize(vmin=min_param, vmax=max_param)
                edge_color = float_to_red_green_color(flattened_edges[edge]) if not colorblind else float_to_red_blue_color(flattened_edges[edge])
        draw_networkx_edges(g, pos, edgelist=[edge], edge_color=edge_color, arrows=False)
        logger.info(f"Drawing edge: {edge} ({edge_color})")

    # Draw legend if requested
    if draw_legend:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add a colorbar
        plt.colorbar(sm, ax=ax, label="Parameter Value")

    # Set the axis and layout
    axis("off")
    tight_layout()

    # Add edge padding
    plt.margins(0.1)

    # Save the graph to an image file
    savefig(filename, dpi=300)
    logger.info(f"Neural Network Graph saved to '{filename}'.")


def get_model_info(model) -> dict:
    """
    Extracts information about a PyTorch
    model's architecture and parameters.

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


def get_tf_model_info(model) -> dict:
    """
    Extracts information about a TensorFlow
    model's architecture and parameters.
    
    Args:
        model: The TensorFlow model to extract information from.

    Returns:
        tuple: A tuple containing the model's architecture and parameters.
    """
    # Get Input & Output Sizes
    model_weights = model.get_weights()
    input_size = model_weights[0].shape[0]
    output_size = model_weights[-1].shape[0]

    # Get Hidden Layer Neuron Counts
    hidden_sizes = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            hidden_sizes.append(layer.units)

    # Return Model Information
    return {
        "num_layers": len(hidden_sizes) + 1,
        "num_nodes": [input_size] + hidden_sizes,
    }


def get_model_params(model, param_type: str = "all") -> dict:
    """
    Returns the weights or biases of the given model.
    """
    # Check if the parameter type is valid
    assert param_type in ["weight", "bias", "all"], f"Invalid parameter type: {param_type}"

    # Extract the parameters
    model_params = model.state_dict()
    
    # Return the requested parameters
    params = {}
    if param_type != "all":
        for name in model_params:
            if param_type in name:
                params[name] = model_params[name]
            else:
                continue
    else:
        for name in model_params:
            params[name] = model_params[name]

    return params


def get_tf_model_params(model, param_type: str = "all") -> dict:
    """
    Returns the weights or biases of the given model.
    
    Args:
        model: The TensorFlow model to extract parameters from.
        param_type (str): The type of parameter to extract.
        
    Returns:
        dict: The extracted parameters.
    """
    # Check if the parameter type is valid
    assert param_type in ["weight", "bias", "all"], f"Invalid parameter type: {param_type}"

    # Extract the parameters
    model_params = model.get_weights()

    # Return the requested parameters
    params = {}
    if param_type != "all":
        if param_type == "weight":
            for params_index in range(0, len(model_params), 2):
                params[f"{params_index}.weight"] = model_params[params_index]
        elif param_type == "bias":
            for params_index in range(1, len(model_params), 2):
                params[f"{params_index}.bias"] = model_params[params_index]
    else:
        for params_index in range(len(model_params)):
            name = "weight" if params_index % 2 == 0 else "bias"
            params[f"{params_index}.{name}"] = model_params[params_index]
            
    return params
