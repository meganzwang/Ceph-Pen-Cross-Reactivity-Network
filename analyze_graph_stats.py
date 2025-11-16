"""
Calculate statistics on molecular graphs: density, sparsity, nodes, edges
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict


def calculate_graph_density(num_nodes, num_edges):
    """
    Calculate graph density.
    
    Density = actual_edges / max_possible_edges
    For undirected graph: max_edges = n*(n-1)/2
    
    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph (undirected, so count each edge once)
    
    Returns:
        Density value between 0 and 1
    """
    if num_nodes <= 1:
        return 0.0
    
    max_edges = num_nodes * (num_nodes - 1) // 2  # Undirected graph
    # Since PyG stores edges as directed (both directions), divide by 2
    actual_edges = num_edges // 2
    
    return actual_edges / max_edges if max_edges > 0 else 0.0


def analyze_single_graph(graph, drug_name):
    """
    Analyze statistics for a single molecular graph.
    
    Args:
        graph: torch_geometric.data.Data object
        drug_name: Name of the drug
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'drug_name': drug_name,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'node_features_dim': graph.x.shape[1] if graph.x is not None else 0,
        'edge_features_dim': graph.edge_attr.shape[1] if graph.edge_attr is not None else 0,
    }
    
    # Calculate density and sparsity
    stats['density'] = calculate_graph_density(stats['num_nodes'], stats['num_edges'])
    stats['sparsity'] = 1.0 - stats['density']
    
    # Average degree (connections per node)
    stats['avg_degree'] = stats['num_edges'] / stats['num_nodes'] if stats['num_nodes'] > 0 else 0
    
    return stats


def analyze_all_graphs(data_path='data/processed_data.pt'):
    """
    Analyze all molecular graphs in the dataset.
    
    Args:
        data_path: Path to processed data file
    
    Returns:
        DataFrame with statistics for all graphs
    """
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    drug_graphs = data['drug_graphs']
    
    print(f"Found {len(drug_graphs)} molecular graphs")
    
    # Analyze each graph
    all_stats = []
    for drug_name, graph in drug_graphs.items():
        stats = analyze_single_graph(graph, drug_name)
        all_stats.append(stats)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_stats)
    
    return df


def print_summary_statistics(df):
    """
    Print summary statistics for all graphs.
    
    Args:
        df: DataFrame with graph statistics
    """
    print("\n" + "="*60)
    print("MOLECULAR GRAPH STATISTICS SUMMARY")
    print("="*60)
    
    # Basic counts
    print(f"Total number of graphs: {len(df)}")
    print(f"Node feature dimension: {df['node_features_dim'].iloc[0]}")
    print(f"Edge feature dimension: {df['edge_features_dim'].iloc[0]}")
    
    print("\n--- NODE STATISTICS ---")
    print(f"Average nodes per graph: {df['num_nodes'].mean():.2f} ± {df['num_nodes'].std():.2f}")
    print(f"Min nodes: {df['num_nodes'].min()}")
    print(f"Max nodes: {df['num_nodes'].max()}")
    print(f"Median nodes: {df['num_nodes'].median():.1f}")
    
    print("\n--- EDGE STATISTICS ---")
    print(f"Average edges per graph: {df['num_edges'].mean():.2f} ± {df['num_edges'].std():.2f}")
    print(f"Min edges: {df['num_edges'].min()}")
    print(f"Max edges: {df['num_edges'].max()}")
    print(f"Median edges: {df['num_edges'].median():.1f}")
    
    print("\n--- DEGREE STATISTICS ---")
    print(f"Average degree per node: {df['avg_degree'].mean():.2f} ± {df['avg_degree'].std():.2f}")
    print(f"Min avg degree: {df['avg_degree'].min():.2f}")
    print(f"Max avg degree: {df['avg_degree'].max():.2f}")
    
    print("\n--- DENSITY/SPARSITY STATISTICS ---")
    print(f"Average density: {df['density'].mean():.4f} ± {df['density'].std():.4f}")
    print(f"Average sparsity: {df['sparsity'].mean():.4f} ± {df['sparsity'].std():.4f}")
    print(f"Min density: {df['density'].min():.4f}")
    print(f"Max density: {df['density'].max():.4f}")
    
    # Interpretation
    print("\n--- INTERPRETATION ---")
    avg_density = df['density'].mean()
    if avg_density < 0.1:
        density_desc = "Very sparse (typical for molecular graphs)"
    elif avg_density < 0.3:
        density_desc = "Moderately sparse"
    elif avg_density < 0.7:
        density_desc = "Moderately dense"
    else:
        density_desc = "Very dense"
    
    print(f"Graph density classification: {density_desc}")
    print(f"These are {'sparse' if avg_density < 0.3 else 'dense'} graphs - ")
    print(f"only {avg_density*100:.2f}% of possible edges are present on average.")
    
    print("\n--- TOP 5 LARGEST GRAPHS (by nodes) ---")
    top_5_nodes = df.nlargest(5, 'num_nodes')[['drug_name', 'num_nodes', 'num_edges', 'density']]
    for _, row in top_5_nodes.iterrows():
        print(f"{row['drug_name']:20s}: {row['num_nodes']:2d} nodes, {row['num_edges']:2d} edges, density={row['density']:.3f}")
    
    print("\n--- TOP 5 SMALLEST GRAPHS (by nodes) ---")
    top_5_small = df.nsmallest(5, 'num_nodes')[['drug_name', 'num_nodes', 'num_edges', 'density']]
    for _, row in top_5_small.iterrows():
        print(f"{row['drug_name']:20s}: {row['num_nodes']:2d} nodes, {row['num_edges']:2d} edges, density={row['density']:.3f}")


def save_detailed_stats(df, output_path='results/graph_statistics.csv'):
    """
    Save detailed statistics to CSV file.
    
    Args:
        df: DataFrame with graph statistics
        output_path: Path to save CSV file
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort by number of nodes for easier reading
    df_sorted = df.sort_values('num_nodes', ascending=False)
    df_sorted.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\n✓ Detailed statistics saved to: {output_path}")


def plot_distributions(df, save_path='plots/graph_statistics.png'):
    """
    Plot distributions of graph statistics.
    
    Args:
        df: DataFrame with graph statistics
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Number of nodes
        axes[0, 0].hist(df['num_nodes'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Number of Nodes (Atoms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Graph Sizes (Nodes)')
        axes[0, 0].grid(alpha=0.3)
        
        # Number of edges
        axes[0, 1].hist(df['num_edges'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Edges (Bonds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Graph Sizes (Edges)')
        axes[0, 1].grid(alpha=0.3)
        
        # Density
        axes[1, 0].hist(df['density'], bins=10, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].set_xlabel('Graph Density')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Graph Density')
        axes[1, 0].grid(alpha=0.3)
        
        # Average degree
        axes[1, 1].hist(df['avg_degree'], bins=10, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_xlabel('Average Degree per Node')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Average Node Degree')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plots saved to: {save_path}")
        plt.close()
        
    except ImportError:
        print("⚠ matplotlib not available - skipping plots")


def main():
    """
    Main function to analyze graph statistics.
    """
    print("Analyzing molecular graph statistics...")
    
    # Analyze all graphs
    df = analyze_all_graphs()
    
    # Print summary
    print_summary_statistics(df)
    
    # Save detailed stats
    save_detailed_stats(df)
    
    # Plot distributions
    plot_distributions(df)
    
    print("\n✓ Graph analysis complete!")
    
    return df


if __name__ == '__main__':
    stats_df = main()
