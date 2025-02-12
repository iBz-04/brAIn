"""
A simple simulation of a human-inspired reflex arc using an integrate-and-fire neuron model.

This simulation creates three neurons:
  1. Sensor Neuron: Receives input (danger signal).
  2. Interneuron: Processes the sensor's signal.
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
        Update the neuron's state for one time step.

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
    """Simulates a biological reflex arc connecting three neurons in sequence.
    
    Attributes:
        sensor: Neuron receiving external danger signals
        interneuron: Neuron processing sensor inputs
        motor: Neuron triggering escape responses
    """
    def __init__(self, 
                 sensor_thresh=1.0, 
                 interneuron_thresh=1.0,
                 motor_thresh=1.0,
                 signal_strength=1.0):
        """
        Args:
            sensor_thresh: Firing threshold for sensor neuron
            interneuron_thresh: Firing threshold for interneuron
            motor_thresh: Firing threshold for motor neuron
            signal_strength: Strength of signals between neurons
        """
        self.signal_strength = signal_strength
        self.sensor = Neuron(threshold=sensor_thresh, decay=0.9)
        self.interneuron = Neuron(threshold=interneuron_thresh, decay=0.95)
        self.motor = Neuron(threshold=motor_thresh, decay=0.95)

    def step(self, danger_signal):
        """Process one time step of the reflex arc.
        
        Args:
            danger_signal: Input value (0.0-1.0) representing danger intensity
            
        Returns:
            Dict containing firing states of all three neurons
        """
        # Clear sensor input from previous step
        self.sensor.membrane_potential *= self.sensor.decay
        
        # Process danger signal through neural chain
        self.sensor.receive_input(danger_signal)
        sensor_fired = self.sensor.update()
        
        # Propagate signal through interneuron if sensor fired
        if sensor_fired:
            self.interneuron.receive_input(self.signal_strength)
        interneuron_fired = self.interneuron.update()
        
        # Trigger motor neuron if interneuron fired
        if interneuron_fired:
            self.motor.receive_input(self.signal_strength)
        motor_fired = self.motor.update()

        return {
            'sensor': sensor_fired,
            'interneuron': interneuron_fired,
            'motor': motor_fired
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
