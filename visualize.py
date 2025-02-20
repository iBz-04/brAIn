import matplotlib.pyplot as plt
from brain import ReflexArc, simulate_reflex_arc
import networkx as nx
import numpy as np

def visualize_simulation(simulation_time: int, danger_times: list) -> None:
    """
    Run the reflex arc simulation and visualize the firing events of each neuron.

    :param simulation_time: Total number of time steps to simulate.
    :param danger_times: List of time step indices when the danger signal is active.
    """
    # Run simulation to get firing events per time step
    simulation_results = simulate_reflex_arc(simulation_time, danger_times)
    
    # Extract firing events for each neuron over time
    sensor_firing = [1 if result['sensor'] else 0 for result in simulation_results]
    interneuron_firing = [1 if result['interneuron'] else 0 for result in simulation_results]
    motor_firing = [1 if result['motor'] else 0 for result in simulation_results]
    
    time_steps = list(range(simulation_time))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot firing events using different markers and line styles for clarity
    plt.plot(time_steps, sensor_firing, marker='o', linestyle='-', label='Sensor Neuron')
    plt.plot(time_steps, interneuron_firing, marker='s', linestyle='-', label='Interneuron')
    plt.plot(time_steps, motor_firing, marker='^', linestyle='-', label='Motor Neuron')
    
    plt.xlabel("Time Step")
    plt.ylabel("Firing Event [1 = Fired, 0 = Not Fired]")
    plt.title("Reflex Arc Simulation - Neuron Firing Visualization")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    
    plt.show()

def visualize_advanced_simulation(simulation_time: int, danger_times: list) -> None:
    """Enhanced visualization showing thresholds and synaptic weights."""
    simulation_results = simulate_reflex_arc(simulation_time, danger_times)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    time_steps = list(range(simulation_time))
    
    # Plot 1: Firing events
    sensor_firing = [1 if result['sensor'] else 0 for result in simulation_results]
    interneuron_firing = [1 if result['interneuron'] else 0 for result in simulation_results]
    motor_firing = [1 if result['motor'] else 0 for result in simulation_results]
    
    ax1.plot(time_steps, sensor_firing, marker='o', label='Sensor')
    ax1.plot(time_steps, interneuron_firing, marker='s', label='Interneuron')
    ax1.plot(time_steps, motor_firing, marker='^', label='Motor')
    ax1.set_title('Neuron Firing Events')
    ax1.legend()
    
    # Plot 2: Adaptive Thresholds
    thresholds_sensor = [result['threshold_sensor'] for result in simulation_results]
    thresholds_inter = [result['threshold_interneuron'] for result in simulation_results]
    thresholds_motor = [result['threshold_motor'] for result in simulation_results]
    
    ax2.plot(time_steps, thresholds_sensor, label='Sensor Threshold')
    ax2.plot(time_steps, thresholds_inter, label='Interneuron Threshold')
    ax2.plot(time_steps, thresholds_motor, label='Motor Threshold')
    ax2.set_title('Adaptive Thresholds')
    ax2.legend()
    
    # Plot 3: Synaptic Weights
    synaptic_strengths = [result['synaptic_strength'] for result in simulation_results]
    ax3.plot(time_steps, synaptic_strengths, label='Synaptic Strength')
    ax3.set_title('Synaptic Weight Evolution')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

class CircuitVisualizer:
    """Provides advanced visualization tools for a NeuralCircuit."""
    def __init__(self, circuit):
        self.circuit = circuit
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        # Add nodes for each neuron
        for i in range(len(self.circuit.neurons)):
            self.G.add_node(i, label=f'Neuron {i}')
        # Add edges for each connection
        for src, targets in self.circuit.connections.items():
            for (tgt, weight) in targets:
                self.G.add_edge(src, tgt, weight=weight)

    def visualize(self):
        """Visualize the neural circuit using an enhanced spring layout with dynamic edge widths."""
        pos = nx.spring_layout(self.G, k=0.5, iterations=50)
        plt.figure(figsize=(10, 10))

        # Draw nodes with updated style
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=700)

        # Draw edges with thickness proportional to weight
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        scaled_widths = [2 * abs(w) for w in weights]
        nx.draw_networkx_edges(self.G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=scaled_widths)

        # Draw node labels with enhanced font size and color
        labels = nx.get_node_attributes(self.G, 'label')
        nx.draw_networkx_labels(self.G, pos, labels, font_size=14, font_color='darkblue')

        # Draw edge labels, showing weights rounded to two decimals
        edge_labels = {(u, v): f'{self.G[u][v]["weight"]:.2f}' for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

        plt.title('Enhanced Neural Circuit Visualization')
        plt.axis('off')
        plt.show()

def visualize_realtime_simulation(simulation_time: int, max_danger: float = 5.0) -> None:
    """Visualization with dynamic danger levels and scaled responses"""
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Initialize data stores with danger levels
    time_steps = []
    sensor_data, danger_levels, motor_data = [], [], []
    
    # Create plot elements with explicit colors
    sensor_line, = ax.plot([], [], 'o-', color='#2ca02c', label='Sensor Activation')  # Green
    danger_bars = ax.bar([], [], color='#ff7f0e', alpha=0.3, label='Danger Level')     # Orange
    motor_line, = ax.plot([], [], '^-', color='#d62728', label='Motor Response')      # Red
    
    # Axis configuration
    ax.set_xlim(0, simulation_time)
    ax.set_ylim(0, max_danger * 1.2)
    ax.set_title("Neuron Danger Response System")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Activation Level / Danger Intensity")
    
    # Create legend with correct colors and markers
    legend_elements = [
        sensor_line,
        plt.Rectangle((0,0), 1, 1, color='#ff7f0e', alpha=0.3),  # Custom rectangle for danger bars
        motor_line
    ]
    ax.legend(legend_elements, ['Sensor Activation', 'Danger Level', 'Motor Response'], 
             loc='upper right', framealpha=0.9)
    
    arc = ReflexArc()
    for t in range(simulation_time):
        # Generate random danger with varying intensity
        current_danger = np.random.rand() * max_danger  # 0 to max_danger
        if np.random.rand() < 0.7: # 70% chance of no danger
            current_danger = 0.0
            
        outputs = arc.step(current_danger)
        
        # Update data stores
        time_steps.append(t)
        sensor_data.append(outputs['sensor'])
        danger_levels.append(current_danger)
        motor_data.append(outputs['motor'])
        
        # Update danger bars
        for rect, height in zip(danger_bars, danger_levels):
            rect.set_height(height)
        danger_bars = ax.bar(time_steps, danger_levels, color='#ff7f0e', alpha=0.3)
        
        # Update lines
        sensor_line.set_data(time_steps, sensor_data)
        motor_line.set_data(time_steps, motor_data)
        
        # Real-time rendering
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.15)
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    visualize_realtime_simulation(simulation_time=50, max_danger=5.0) 