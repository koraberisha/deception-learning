"""
Visualization utilities for training metrics.

This module provides interactive visualization tools for model training metrics 
that work on both local and remote servers.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
from IPython.display import display, HTML

class TrainingVisualizer:
    """
    Class for visualizing training metrics from saved JSON files.
    """
    
    def __init__(self, metrics_dir):
        """
        Initialize the visualizer.
        
        Args:
            metrics_dir: Directory containing metrics files
        """
        self.metrics_dir = metrics_dir
        self.metrics_file = os.path.join(metrics_dir, "training_metrics.json")
        self.loss_history_file = os.path.join(metrics_dir, "loss_history.json")
        self.step_loss_file = os.path.join(metrics_dir, "step_losses.json")
        
        # Load metrics data if available
        self.metrics_data = self._load_metrics()
        self.step_losses = self._load_step_losses()
        self.loss_history = self._load_loss_history()
    
    def _load_metrics(self):
        """Load the metrics from the main metrics file."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {"loss": None, "metrics": {}}
    
    def _load_step_losses(self):
        """Load step-by-step losses if available."""
        if os.path.exists(self.step_loss_file):
            with open(self.step_loss_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_loss_history(self):
        """Load the loss history if available."""
        if os.path.exists(self.loss_history_file):
            with open(self.loss_history_file, 'r') as f:
                return json.load(f)
        return {"steps": [], "losses": []}
    
    def plot_loss_curve(self, save_path=None, show=True):
        """
        Plot the loss curve from the loss history.
        
        Args:
            save_path: Path to save the plot image
            show: Whether to display the plot
        """
        if not self.loss_history["steps"]:
            print("No loss history data available.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history["steps"], self.loss_history["losses"], marker='o', linestyle='-', markersize=4)
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure x-axis has integer ticks for steps
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a trend line
        if len(self.loss_history["steps"]) > 1:
            z = np.polyfit(self.loss_history["steps"], self.loss_history["losses"], 1)
            p = np.poly1d(z)
            plt.plot(self.loss_history["steps"], p(self.loss_history["steps"]), "r--", alpha=0.7, 
                    label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
            plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss curve saved to {save_path}")
        
        if show:
            plt.tight_layout()
            plt.show()
    
    def plot_reward_components(self, save_path=None, show=True):
        """
        Plot the various reward components if available.
        
        Args:
            save_path: Path to save the plot image
            show: Whether to display the plot
        """
        # Check if we have step losses with reward components
        if not self.step_losses:
            print("No step-level reward data available.")
            return
        
        # Extract reward components across steps
        steps = []
        reward_components = {}
        
        for step, data in self.step_losses.items():
            if 'reward_components' in data:
                steps.append(int(step))
                for reward_name, value in data['reward_components'].items():
                    if reward_name not in reward_components:
                        reward_components[reward_name] = []
                    reward_components[reward_name].append(value)
        
        if not reward_components:
            print("No reward component data found in step losses.")
            return
        
        # Plot the reward components
        plt.figure(figsize=(12, 8))
        for reward_name, values in reward_components.items():
            if len(steps) == len(values):  # Ensure data alignment
                plt.plot(steps, values, marker='o', linestyle='-', markersize=3, label=reward_name)
        
        plt.title('Reward Components Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Reward Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Reward components plot saved to {save_path}")
        
        if show:
            plt.tight_layout()
            plt.show()


import os
import datetime


def generate_html_report(self, output_file='training_report.html'):
    """
    Generate an HTML report with all visualizations.

    Args:
        output_file: Name of the HTML output file
    """
    loss_plot_path = os.path.join(self.metrics_dir, 'loss_curve.png')
    reward_plot_path = os.path.join(self.metrics_dir, 'reward_components.png')

    # Generate the plots
    self.plot_loss_curve(save_path=loss_plot_path, show=False)
    self.plot_reward_components(save_path=reward_plot_path, show=False)

    # Get current time as a string
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .metrics {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Training Report</h1>
                <p>Generated on {current_time}</p>
            </div>

            <div class="plot">
                <h2>Training Loss</h2>
                <img src="loss_curve.png" alt="Training Loss Curve">
            </div>

            <div class="plot">
                <h2>Reward Components</h2>
                <img src="reward_components.png" alt="Reward Components">
            </div>

            <div class="metrics">
                <h2>Training Metrics Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Final Loss</td>
                        <td>{self.metrics_data.get("loss", "N/A")}</td>
                    </tr>
    """

    # Add other metrics from the metrics data
    for key, value in self.metrics_data.get("metrics", {}).items():
        html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    # Write the HTML file
    output_path = os.path.join(self.metrics_dir, output_file)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report generated at {output_path}")
    return output_path


def update_loss_history(metrics_dir, step, loss):
    """
    Update the loss history file with a new step and loss value.
    
    Args:
        metrics_dir: Directory to save metrics
        step: Current training step
        loss: Loss value for this step
    """
    history_file = os.path.join(metrics_dir, "loss_history.json")
    
    # Load existing history or create a new one
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {"steps": [], "losses": []}
    
    # Update with new data
    history["steps"].append(step)
    history["losses"].append(loss)
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def update_step_loss(metrics_dir, step, loss_data):
    """
    Update the step loss file with detailed loss information.
    
    Args:
        metrics_dir: Directory to save metrics
        step: Current training step
        loss_data: Dictionary with loss details for this step
    """
    step_loss_file = os.path.join(metrics_dir, "step_losses.json")
    
    # Load existing data or create a new one
    if os.path.exists(step_loss_file):
        with open(step_loss_file, 'r') as f:
            step_losses = json.load(f)
    else:
        step_losses = {}
    
    # Update with new data
    step_losses[str(step)] = loss_data
    
    # Save updated data
    with open(step_loss_file, 'w') as f:
        json.dump(step_losses, f, indent=2)

def setup_interactive_visualization(metrics_dir):
    """
    Set up interactive visualization for metrics using IPython display.
    This works in Jupyter notebooks.
    
    Args:
        metrics_dir: Directory containing metrics files
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            print("Interactive visualization requires running in a Jupyter notebook.")
            return
        
        visualizer = TrainingVisualizer(metrics_dir)
        
        # Display interactive visualizations
        print("Setting up interactive visualizations...")
        
        # Display loss curve
        plt.figure(figsize=(10, 6))
        visualizer.plot_loss_curve(show=False)
        plt.tight_layout()
        display(plt.gcf())
        plt.close()
        
        # Display reward components
        plt.figure(figsize=(12, 8))
        visualizer.plot_reward_components(show=False)
        plt.tight_layout()
        display(plt.gcf())
        plt.close()
        
    except ImportError:
        print("Interactive visualization requires IPython and matplotlib.")

def main():
    """Main function for command-line use."""
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--metrics_dir", type=str, default="outputs", 
                        help="Directory containing metrics files")
    parser.add_argument("--output", type=str, default="training_report.html",
                        help="Output HTML report filename")
    parser.add_argument("--plot_loss", action="store_true", 
                        help="Generate and display loss curve plot")
    parser.add_argument("--plot_rewards", action="store_true",
                        help="Generate and display reward components plot")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to files")
    parser.add_argument("--interactive", action="store_true",
                        help="Set up interactive visualization (for Jupyter)")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML report with all visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TrainingVisualizer(args.metrics_dir)
    
    # Process requested actions
    if args.plot_loss:
        save_path = os.path.join(args.metrics_dir, 'loss_curve.png') if args.save_plots else None
        visualizer.plot_loss_curve(save_path=save_path)
    
    if args.plot_rewards:
        save_path = os.path.join(args.metrics_dir, 'reward_components.png') if args.save_plots else None
        visualizer.plot_reward_components(save_path=save_path)
    
    if args.interactive:
        setup_interactive_visualization(args.metrics_dir)
    
    if args.report:
        visualizer.generate_html_report(args.output)

if __name__ == "__main__":
    main()