import unittest
from brain import Neuron, ReflexArc, simulate_reflex_arc

class TestNeuron(unittest.TestCase):
    def test_neuron_firing(self):
        # Create a neuron with a threshold of 1.0 and no decay for predictability
        neuron: Neuron = Neuron(threshold=1.0, decay=1.0, refractory_period=1)
        # Provide input below threshold; should not fire
        neuron.receive_input(0.5)
        self.assertFalse(neuron.update(), "Neuron should not fire yet when below threshold")
        # Provide additional input to cross the threshold; should fire
        neuron.receive_input(0.6)
        self.assertTrue(neuron.update(), "Neuron should fire when threshold exceeded")
    
    def test_neuron_refractory(self):
        # Set refractory period to 2 time steps
        neuron: Neuron = Neuron(threshold=1.0, decay=1.0, refractory_period=2)
        neuron.receive_input(1.0)
        self.assertTrue(neuron.update(), "Neuron should fire when threshold is met")
        # Immediately after firing, neuron is in refractory period
        neuron.receive_input(1.0)
        self.assertFalse(neuron.update(), "Neuron in refractory period should not fire")
        # After refractory period, provide sufficient input to fire again
        neuron.receive_input(1.0)
        self.assertTrue(neuron.update(), "Neuron should fire after refractory period expires")

class TestReflexArc(unittest.TestCase):
    def test_reflex_arc_single_trigger(self):
        arc: ReflexArc = ReflexArc()
        # Trigger the sensor neuron directly by providing a danger input of 1.0
        outputs = arc.step(1.0)
        self.assertTrue(outputs['sensor'], "Sensor neuron should fire with danger input")
        # If sensor fires, the interneuron should receive input, and possibly fire
        self.assertIn(outputs['interneuron'], [True, False], "Interneuron output must be boolean")
        # Similarly for the motor neuron
        self.assertIn(outputs['motor'], [True, False], "Motor output must be boolean")

    def test_simulation_run(self):
        # Run a short simulation with a known danger time
        simulation_results = simulate_reflex_arc(simulation_time=5, danger_times=[2])
        # Ensure there are 5 time steps of output
        self.assertEqual(len(simulation_results), 5, "Simulation should produce 5 time steps of results")
        # Check that at time 2, the sensor fired
        self.assertTrue(simulation_results[2]['sensor'], "Sensor neuron should fire when danger input is provided")

if __name__ == '__main__':
    unittest.main() 