
"""
A simple simulation of a human-inspired reflex arc using an integrate-and-fire neuron model.

This simulation creates three neurons:
  1. Sensor Neuron: Receives input (danger signal).
  2. Interneuron: Processes the sensor’s signal.
  3. Motor Neuron: Triggers an escape response when activated.

Each neuron integrates input over discrete time steps, applies a decay to mimic the loss of potential,
and fires when its membrane potential exceeds a set threshold. After firing, the neuron resets and enters
a brief refractory period where it cannot fire.
"""


class Neuron:
    def __init__(self, threshold, decay=0.95, refractory_period=2):
        """
        Initialize the neuron.

        :param threshold: The membrane potential needed for the neuron to fire.
        :param decay: The factor by which the membrane potential decays each time step.
        :param refractory_period: Number of time steps after firing during which the neuron is inactive.
        """
        self.threshold = threshold  # Firing threshold
        self.decay = decay  # Decay rate per time step
        self.refractory_period = refractory_period  # Time steps the neuron remains inactive after firing
        self.membrane_potential = 0.0  # Current integrated potential
        self.refractory_timer = 0  # Counter for refractory period

    def receive_input(self, input_value):
        """
        Add input to the neuron's membrane potential if not in refractory.

        :param input_value: The value to add to the membrane potential.
        """
        if self.refractory_timer <= 0:
            self.membrane_potential += input_value

    def update(self):
        """
        Update the neuron’s state for one time step.

        Applies decay to the membrane potential, checks for firing, and manages the refractory timer.

        :return: True if the neuron fires on this time step, False otherwise.
        """
        fired = False
        if self.refractory_timer > 0:
            # Decrease refractory counter if neuron is inactive
            self.refractory_timer -= 1
        else:
            if self.membrane_potential >= self.threshold:
                fired = True
                # Reset the potential and set the refractory period upon firing
                self.membrane_potential = 0.0
                self.refractory_timer = self.refractory_period
            else:
                # Apply decay if not firing
                self.membrane_potential *= self.decay
        return fired


class ReflexArc:
    """
    Represents a basic reflex arc with a sensor neuron, an interneuron, and a motor neuron.
    """

    def __init__(self):
        # Initialize neurons with parameters chosen to mimic a fast reflex
        self.sensor = Neuron(threshold=1.0, decay=0.9, refractory_period=1)
        self.interneuron = Neuron(threshold=1.0, decay=0.95, refractory_period=1)
        self.motor = Neuron(threshold=1.0, decay=0.95, refractory_period=1)

    def step(self, danger_signal):
        """
        Process one simulation time step.

        :param danger_signal: A float representing danger intensity (e.g., 0.0 for safe, 1.0 for danger).
        :return: Dictionary with the firing state of each neuron.
        """
        # Sensor neuron receives the danger input
        self.sensor.receive_input(danger_signal)
        sensor_fired = self.sensor.update()

        # If sensor fires, pass a signal to the interneuron
        if sensor_fired:
            self.interneuron.receive_input(1.0)  # Fixed signal strength for simplicity

        interneuron_fired = self.interneuron.update()

        # If interneuron fires, pass the signal to the motor neuron
        if interneuron_fired:
            self.motor.receive_input(1.0)

        motor_fired = self.motor.update()

        return {
            'sensor': sensor_fired,
            'interneuron': interneuron_fired,
            'motor': motor_fired,
        }


def simulate_reflex_arc(simulation_time, danger_times):
    """
    Run the simulation over a given number of time steps.

    :param simulation_time: Total number of time steps to simulate.
    :param danger_times: List of time step indices when the danger signal is active.
    :return: A list of dictionaries containing the state of each neuron per time step.
    """
    arc = ReflexArc()
    simulation_results = []

    for t in range(simulation_time):
        # Provide danger input only on specified time steps
        danger_signal = 1.0 if t in danger_times else 0.0
        outputs = arc.step(danger_signal)
        simulation_results.append(outputs)
        print(f"Time {t}: Danger input: {danger_signal} | "
              f"Sensor fired: {outputs['sensor']} | "
              f"Interneuron fired: {outputs['interneuron']} | "
              f"Motor fired: {outputs['motor']}")

    return simulation_results


if __name__ == '__main__':
    # simulate 20 time steps with danger signals at time steps 3 and 10
    simulate_reflex_arc(simulation_time=20, danger_times=[3, 10])
