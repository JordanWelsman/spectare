# Define a dictionary mapping numbers to their subscript/superscript equivalents
subscript_map = {
    '0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083',
    '4': '\u2084', '5': '\u2085', '6': '\u2086', '7': '\u2087',
    '8': '\u2088', '9': '\u2089'
}

# Define a dictionary mapping numbers to their superscript equivalents
superscript_map = {
    '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
    '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
    '8': '\u2078', '9': '\u2079'
}

def node_notation(layer, node) -> str:
    """
    Return a formatted string representing
    the node in the given layer in node
    notation (a^{[l]}_{h}).

    Parameters:
        layer (int): The layer index
        node (int): The node index

    Returns:
        str: The formatted node notation

    Examples:
        >>> node_notation(0, 0)
        'a₁⁰'
        >>> node_notation(6, 12)
        'a₇¹²'
    """
    return f"a{subscript_map[str(layer + 1)]}{superscript_map[str(node)]}"

print(node_notation(3, 6))