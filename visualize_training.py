#!/usr/bin/env python3
"""
Script to visualize training metrics for the Secret Conversations model.

This script can be run during or after training to see real-time progress
and analyze model training metrics.
"""

import argparse
import os
from utils.visualization import TrainingVisualizer, setup_interactive_visualization

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--metrics_dir", type=str, default="outputs",
                        help="Directory containing metrics files")
    parser.add_argument("--output", type=str, default="training_report.html",
                        help="Output HTML report filename")
    parser.add_argument("--interactive", action="store_true",
                        help="Use interactive mode (for Jupyter notebooks)")
    parser.add_argument("--mode", choices=["report", "plot", "all"], default="all",
                        help="Visualization mode: report, plot, or all")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if metrics directory exists
    if not os.path.exists(args.metrics_dir):
        print(f"Error: Metrics directory '{args.metrics_dir}' not found.")
        return
    
    # Create visualizer
    visualizer = TrainingVisualizer(args.metrics_dir)
    
    if args.interactive:
        print("Setting up interactive visualization...")
        setup_interactive_visualization(args.metrics_dir)
        return
    
    if args.mode in ["plot", "all"]:
        # Plot loss curve
        print("Generating loss curve...")
        visualizer.plot_loss_curve(save_path=os.path.join(args.metrics_dir, "loss_curve.png"), show=True)
        
        # Plot reward components
        print("Generating reward components plot...")
        visualizer.plot_reward_components(save_path=os.path.join(args.metrics_dir, "reward_components.png"), show=True)
    
    if args.mode in ["report", "all"]:
        # Generate HTML report
        print("Generating HTML report...")
        report_path = visualizer.generate_html_report(args.output)
        print(f"Report generated at: {report_path}")
        
        # Print how to view the report on remote server
        print("\nTo view the report on a remote server, you can:")
        print("1. Start a local web server:")
        print(f"   python -m http.server --directory {args.metrics_dir} 8000")
        print("2. Access from your local machine through SSH port forwarding:")
        print("   ssh -L 8000:localhost:8000 your_remote_server")
        print("3. Then open in your browser: http://localhost:8000/training_report.html")

if __name__ == "__main__":
    main()