"""
This script extracts graph data from an HTML file containing vis.js visualizations,
converts it to DOT format, and then generates a PDF using Graphviz's sfdp tool.
Used to create visualizations of graphs for manuscript.
HTML files have to be generated first using src/data/dbpedia_database_visualize.py
"""

import json
import re
import subprocess
import os
import sys

def extract_json_from_text(text, start_pattern):
    start_match = re.search(start_pattern, text)
    if not start_match:
        return None, f"Start pattern '{start_pattern}' not found"
    
    start_index = start_match.end()
    # Find the start of the array
    array_start = text.find('[', start_index)
    if array_start == -1:
         return None, "Opening bracket '[' not found after pattern"
         
    # Simple bracket counting to find the matching closing bracket
    count = 0
    for i in range(array_start, len(text)):
        if text[i] == '[':
            count += 1
        elif text[i] == ']':
            count -= 1
            if count == 0:
                return text[array_start:i+1], None
                
    return None, "Matching closing bracket ']' not found"

def extract_data(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"File read. Size: {len(content)} chars")

    # Extract nodes
    nodes_json_str, err = extract_json_from_text(content, r'nodes\s*=\s*new\s*vis\.DataSet\(')
    if not nodes_json_str:
        print(f"Error extracting nodes: {err}")
        return None, None
        
    # Extract edges
    edges_json_str, err = extract_json_from_text(content, r'edges\s*=\s*new\s*vis\.DataSet\(')
    if not edges_json_str:
         print(f"Error extracting edges: {err}")
         return None, None

    try:
        nodes = json.loads(nodes_json_str)
        edges = json.loads(edges_json_str)
        print(f"Successfully parsed {len(nodes)} nodes and {len(edges)} edges.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Print a snippet of the failing string
        print(f"Snippet: {nodes_json_str[:100]} ...")
        return None, None

    return nodes, edges

def generate_dot(nodes, edges, output_dot_path):
    dot_content = "digraph G {\n"
    # Force-directed layout settings
    dot_content += "  layout=sfdp;\n"
    dot_content += "  K=0.00075;\n"  
    dot_content += "  overlap=prism;\n"
    dot_content += "  splines=true;\n"
    dot_content += "  node [shape=ellipse, style=filled, color=lightblue, fontname=\"Helvetica\", fontsize=24];\n"
    dot_content += "  edge [fontname=\"Helvetica\", fontsize=16];\n"
    
    # Add nodes
    # First, determine leaf nodes (nodes with no outgoing edges)
    from_node_ids = {str(edge.get('from')) for edge in edges}
    
    for node in nodes:
        node_id = str(node.get('id'))
        label = node.get('label', node_id)
        label = label.replace('"', '\\"') # Escape quotes in label
        
        # Assign a lighter color for leaf nodes
        if node_id not in from_node_ids:
            fillcolor = "#cfe2f3" 
        else:
            fillcolor = node.get('color', '#97c2fc') 

        dot_content += f'  "{node_id}" [label="{label}", fillcolor="{fillcolor}"];\n'

    # Add edges
    for edge in edges:
        from_node = str(edge.get('from'))
        to_node = str(edge.get('to'))
        dot_content += f'  "{from_node}" -> "{to_node}";\n'

    dot_content += "}\n"

    with open(output_dot_path, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    print(f"DOT file written to {output_dot_path}")

def convert_to_pdf(dot_path, pdf_path):
    try:
        # Use sfdp for force-directed layout 
        subprocess.run(['sfdp', '-Tpdf', dot_path, '-o', pdf_path], check=True)
        print(f"PDF file generated at {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running sfdp command: {e}")
    except FileNotFoundError:
        print("Error: 'sfdp' command not found. Please install graphviz.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_and_convert.py <input_html_file>")
        sys.exit(1)
        
    html_file = sys.argv[1]
    
    # Generate output file names based on input HTML file name
    base_name = os.path.splitext(os.path.basename(html_file))[0]
    dot_file = f"{base_name}.dot"
    pdf_file = f"{base_name}.pdf"
    
    nodes, edges = extract_data(html_file)
    if nodes and edges:
        generate_dot(nodes, edges, dot_file)
        convert_to_pdf(dot_file, pdf_file)
