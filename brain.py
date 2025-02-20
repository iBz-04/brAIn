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

import numpy as np
import yaml

class Neuron:
    def __init__(self, threshold, decay=0.95, refractory_period=2, adaptation_rate=0.1, 
                 metabolic_rate=0.1, energy_capacity=100.0):
        """
        Initialize the neuron with adaptive properties.

        :param threshold: The membrane potential needed for the neuron to fire.
        :param decay: The factor by which the membrane potential decays each time step.
        :param refractory_period: Number of time steps after firing during which the neuron is inactive.
        :param adaptation_rate: Rate at which the threshold adapts based on recent firing activity.
        :param metabolic_rate: ATP consumption per ms
        :param energy_capacity: Maximum energy reserve
        """
        self.base_threshold = threshold
        self.threshold = threshold
        self.decay = decay  # Decay rate per time step
        self.refractory_period = refractory_period  # Time steps the neuron remains inactive after firing
        self.membrane_potential = 0.0  # Current integrated potential
        self.refractory_timer = 0  # Counter for refractory period
        self.synaptic_weights = 1.0  # Initial synaptic weight
        
        # New adaptive parameters
        self.adaptation_rate = adaptation_rate
        self.spike_count = 0
        self.last_spike_time = 0
        self.firing_history = []
        
        # Energy-related parameters
        self.metabolic_rate = metabolic_rate
        self.energy = energy_capacity
        self.energy_capacity = energy_capacity
        self.glial_support = 0.1  # Energy replenishment rate

    def adapt_threshold(self, current_time):
        """Adapt threshold based on recent firing activity."""
        if len(self.firing_history) > 0:
            recent_activity = sum(1 for t in self.firing_history 
                                if current_time - t < 10)  # Look at last 10 time steps
            # Increase threshold if firing frequently, decrease if rarely firing
            self.threshold = self.base_threshold * (1 + self.adaptation_rate * recent_activity)

    def receive_input(self, input_value):
        """
        Add input to the neuron's membrane potential if not in refractory.

        :param input_value: The value to add to the membrane potential.
        """
        if self.refractory_timer <= 0:
            self.membrane_potential += input_value

    def update_energy(self, dt):
        """Manage energy metabolism with glial support"""
        self.energy = min(
            self.energy_capacity,
            self.energy - self.metabolic_rate * dt + self.glial_support * dt
        )
        
    def can_fire(self):
        """Check energy requirements for spike generation"""
        return self.energy > self.metabolic_rate * 5  # Minimum reserve for spike

    def update(self, current_time=0):
        """
        Update the neuron's state for one time step.

        Applies decay to the membrane potential, checks for firing, and manages the refractory timer.

        :return: True if the neuron fires on this time step, False otherwise.
        """
        if not self.can_fire():
            return False
            
        fired = False
        
        # Adapt threshold based on recent activity
        self.adapt_threshold(current_time)
        
        if self.refractory_timer > 0:
            # Decrease refractory counter if neuron is inactive
            self.refractory_timer -= 1
        else:
            if self.membrane_potential >= self.threshold:
                fired = True
                # Reset the potential and set the refractory period upon firing
                self.membrane_potential = 0.0
                self.refractory_timer = self.refractory_period
                self.firing_history.append(current_time)
                self.spike_count += 1
                self.energy -= self.metabolic_rate * 2  # Spike energy cost
            else:
                # Apply decay if not firing
                self.membrane_potential *= self.decay
        return fired

    def apply_synaptic_plasticity(self, pre_spike_time, post_spike_time):
        """
        Update synaptic weights based on the timing difference between
        pre-synaptic and post-synaptic spikes using a simplified STDP rule.

        If the pre-synaptic spike occurs before the post-synaptic spike,
        potentiation occurs; otherwise, depression occurs.
        """
        delta_t = post_spike_time - pre_spike_time
        if delta_t > 0:
            # Potentiation: increase the synaptic weight
            self.synaptic_weights *= (1 + 0.05 * abs(delta_t))
        else:
            # Depression: decrease the synaptic weight
            self.synaptic_weights *= (1 - 0.05 * abs(delta_t))


class NeuralCircuit:
    """Configurable neural network with structural plasticity"""
    def __init__(self, config_file=None):
        self.neurons = {}
        self.connections = {}
        self.plasticity_rules = []
        self.metabolic_rate = 0.0
        self.energy_pool = 100.0  # Initial energy reserve
        
        if config_file:
            self.load_configuration(config_file)

    def load_configuration(self, config_path: str):
        """Load neural circuit configuration from YAML/JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # Load neural components
                self._parse_neurons(config.get('neurons', {}))
                self._parse_connections(config.get('connections', []))
                self._parse_plasticity_rules(config.get('plasticity_rules', []))
                
                # Load energy parameters
                energy_config = config.get('energy', {})
                self.energy_pool = energy_config.get('initial_pool', 100.0)
                self.metabolic_rate = energy_config.get('replenishment_rate', 0.5)
                
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file {config_path} not found") from exc
        except yaml.YAMLError as exc:
            raise RuntimeError(f"YAML parsing error in {config_path}") from exc
        except KeyError as exc:
            raise RuntimeError(f"Missing required configuration key: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error loading configuration: {str(exc)}") from exc

    def _parse_neurons(self, neurons_config: dict):
        """Parse neurons from configuration"""
        for neuron_id, spec in neurons_config.items():
            self.add_neuron(
                neuron_id=neuron_id,
                neuron_type=spec['type'],
                params=spec.get('params', {})
            )

    def _parse_connections(self, connections_config: list):
        """Parse neural connections from configuration"""
        for conn in connections_config:
            if len(conn) == 3:
                self.add_connection(conn[0], conn[1], conn[2])

    def _parse_plasticity_rules(self, rules_config: list):
        """Parse plasticity rules from configuration"""
        for rule in rules_config:
            self.plasticity_rules.append({
                'type': rule['type'],
                'parameters': rule.get('parameters', {})
            })

    def add_neuron(self, neuron_id, neuron_type, params):
        """Add a neuron to the circuit with initial parameters"""
        self.neurons[neuron_id] = {
            'type': neuron_type,
            'params': params,
            'energy': 100.0,
            'activity': 0.0
        }

    def add_connection(self, source, target, initial_weight):
        """Create a mutable connection between neurons"""
        if source not in self.connections:
            self.connections[source] = []
        self.connections[source].append({
            'target': target,
            'weight': initial_weight,
            'activity': 0.0
        })

    def structural_plasticity(self):
        """Dynamic connection modification based on activity"""
        for source in list(self.connections.keys()):
            for connection in self.connections[source]:
                # Hebbian rule implementation
                if self.neurons[source]['activity'] > 0.5 and self.neurons[connection['target']]['activity'] > 0.5:
                    connection['weight'] = min(connection['weight'] * 1.1, 2.0)
                # Prune unused connections
                if connection['activity'] < 0.01:
                    self.connections[source].remove(connection)
                    
    def metabolic_update(self):
        """Manage energy consumption and recovery"""
        energy_cost = sum(neuron['params']['metabolic_rate'] for neuron in self.neurons.values())
        self.energy_pool = max(0, self.energy_pool - energy_cost)
        
        for neuron in self.neurons.values():
            neuron['energy'] = max(0, neuron['energy'] - neuron['params']['metabolic_rate'])


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
                 signal_strength=1.0,
                 learning_rate=0.01):
        """Initialize with learning capabilities."""
        self.signal_strength = signal_strength
        self.learning_rate = learning_rate
        self.current_time = 0
        
        # Initialize neurons with adaptive thresholds
        self.sensor = Neuron(threshold=sensor_thresh, decay=0.9)
        self.interneuron = Neuron(threshold=interneuron_thresh, decay=0.95)
        self.motor = Neuron(threshold=motor_thresh, decay=0.95)
        
        # Track spike timing for STDP
        self.last_spike_times = {
            'sensor': 0,
            'interneuron': 0,
            'motor': 0
        }

    def update_synaptic_weights(self):
        """Update connection strengths based on spike timing."""
        # Update sensor -> interneuron connection
        if self.last_spike_times['interneuron'] > self.last_spike_times['sensor']:
            # Strengthen connection if interneuron fires after sensor
            time_diff = self.last_spike_times['interneuron'] - self.last_spike_times['sensor']
            self.signal_strength *= (1 + self.learning_rate * np.exp(-time_diff))
        
    def step(self, danger_level):
        """Process danger levels from 0.0 to 5.0"""
        self.current_time += 1
        
        # Scale input based on danger level (0-5 -> 0-1.5 input multiplier)
        scaled_input = danger_level / 3.3  
        self.sensor.receive_input(scaled_input)
        
        # Process danger signal through neural chain
        sensor_fired = self.sensor.update(self.current_time)
        
        if sensor_fired:
            self.last_spike_times['sensor'] = self.current_time
            self.interneuron.receive_input(self.signal_strength)
            
        interneuron_fired = self.interneuron.update(self.current_time)
        
        if interneuron_fired:
            self.last_spike_times['interneuron'] = self.current_time
            self.motor.receive_input(self.signal_strength)
            
        motor_fired = self.motor.update(self.current_time)
        
        if motor_fired:
            self.last_spike_times['motor'] = self.current_time
        
        # Update synaptic weights based on spike timing
        self.update_synaptic_weights()
        
        return {
            'sensor': sensor_fired,
            'interneuron': interneuron_fired,
            'motor': motor_fired,
            'threshold_sensor': self.sensor.threshold,
            'threshold_interneuron': self.interneuron.threshold,
            'threshold_motor': self.motor.threshold,
            'synaptic_strength': self.signal_strength
        }


def simulate_reflex_arc(simulation_time, danger_levels):
    """
    Run simulation with varying danger levels.
    
    :param danger_levels: Dictionary {time_step: danger_level} 
           where danger_level is 0.0 (safe) to 5.0 (extreme danger)
    """
    arc = ReflexArc()
    simulation_results = []

    for t in range(simulation_time):
        danger = danger_levels.get(t, 0.0)
        outputs = arc.step(danger)
        simulation_results.append(outputs)
        print(f"Time {t}: Danger level: {danger:.2f} | "
              f"Sensor: {outputs['sensor']} | "
              f"Interneuron: {outputs['interneuron']} | "
              f"Motor: {outputs['motor']}")
    
    return simulation_results


class ContinuousTimeEngine:
    """Event-driven neural simulation with variable time steps"""
    def __init__(self, circuit):
        self.circuit = circuit
        self.event_queue = []
        self.current_time = 0.0
        self.time_step = 0.1  # Initial time resolution in ms

    def process_event(self, event):
        """Handle spike propagation and synaptic integration"""
        neuron = self.circuit.neurons[event['target']]
        dt = event['time'] - self.current_time
        
        # Update membrane potential using Izhikevich model
        dv = (0.04 * neuron['v']**2 + 5 * neuron['v'] + 140 - neuron['u'] + event['current']) 
        neuron['v'] += dv * dt
        
        # Check for spike threshold
        if neuron['v'] >= 30:
            self.generate_spike(neuron, event['time'])
            
    def izhikevich_update(self, params, dt):
        """Izhikevich neuron model differential equations"""
        dv = (0.04 * params['v']**2 + 5 * params['v'] + 140 - params['u'] + params['I'])
        du = params['a'] * (params['b'] * params['v'] - params['u'])
        return dv * dt, du * dt

    def generate_spike(self, neuron, spike_time):
        """Handle spike generation and propagate to connected neurons"""
        neuron['v'] = neuron['params']['c']  # Reset to resting potential
        neuron['u'] += neuron['params']['d']  # Reset recovery variable
        
        # Schedule spike propagation to connected neurons
        for connection in self.circuit.connections.get(neuron['id'], []):
            self.event_queue.append({
                'time': spike_time + 1.0,  # 1ms synaptic delay
                'target': connection['target'],
                'current': connection['weight'] * neuron['synaptic_weights']
            })


if __name__ == '__main__':
    from visualize import visualize_realtime_simulation
    visualize_realtime_simulation(simulation_time=100, max_danger=5.0)
