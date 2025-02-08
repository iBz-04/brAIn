import matplotlib.pyplot as plt
from brain import simulate_reflex_arc

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

if __name__ == "__main__":
    # Adjust simulation_time and danger_times as needed
    visualize_simulation(simulation_time=20, danger_times=[3, 10]) 