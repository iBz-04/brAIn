# br-AI-n

This project is the beginning of a new type of artificial intelligence—one that is not based on statistical learning but on biological principles. Instead of using artificial neural networks trained on data, my goal is to build AI that functions like a human brain, neuron by neuron, system by system, everything hard coded. The approach is questionable but hey, let's try.

The current phase starts small, with the simplest form of neural processing: the reflex arc. This is the foundation of all higher intelligence, as even complex thought processes are built upon basic neural circuits like these.
png
<img src="https://res.cloudinary.com/diekemzs9/image/upload/v1740075929/final_plot_wlngpq.png" alt="Neuron" width="800"/>

At its core, our system consists of:
- **Biologically accurate neurons** that have real-time signal processing, thresholds, decay rates, and refractory periods.
- **A structured nervous system** that processes signals in a way similar to how a human nervous system would.
- **A core task (escaping danger)** as the first "instinct" of our artificial brain, before we add more cognitive functions.

The end goal is not just to create AI that "predicts" based on patterns but AI that thinks, adapts, and functions like a living brain—not through pre-programmed knowledge but through organic learning and decision-making.

## Recent Updates

### Implemented Biological Features
- **Adaptive Thresholds:** Neurons now adjust their firing thresholds based on recent activity (higher activity = higher threshold)
- **Synaptic Plasticity:** Implements spike-timing-dependent plasticity (STDP) for connection strengthening/weakening
- **Advanced Monitoring:** Track threshold adaptations and synaptic weight evolution in real-time
- **Circuit Visualization:** New network visualization showing neuron connections and connection strengths

## Current Code

### Enhanced Components

- **brain.py**  
  Contains the biologically-inspired neural simulation:
  - `Neuron` class - Implements:
    - Real-time membrane potential integration
    - Adaptive firing thresholds
    - Refractory periods
    - Spike-timing tracking
  - `ReflexArc` class - Manages:
    - Sensor-Interneuron-Motor neural chain
    - Spike-timing-dependent plasticity (STDP)
    - Real-time synaptic weight adjustments

- **visualize.py**  
  Enhanced visualization tools:
  - **Firing Events Timeline:** Tracks neuron activation patterns
  - **Adaptive Threshold Plots:** Shows threshold changes over time
  - **Synaptic Evolution Graph:** Visualizes connection strength changes
  - **Circuit Network Diagram:** Displays neural connections with weight-based edge thickness

- **test_brain.py**  
  Expanded test suite covering:
  - Threshold adaptation mechanisms
  - Refractory period compliance
  - STDP weight adjustment rules
  - Multi-neuron signal propagation

## Requirements

- Python 3.x (tested on Python 3.12)
- PyYAML (for configuration files)

## Updated Roadmap

### Next Enhancements

- **Dynamic Neural Networks** (In Progress)
  - Create configurable neural circuits
  - Implement excitatory/inhibitory neuron types
  
- **Advanced Learning Rules**
  - Add long-term potentiation/depression (LTP/LTD)
  - Implement neuromodulator-inspired learning

- **3D Visualization**  
  - Interactive 3D neural activity visualization
  - Real-time potential threshold displays

- **Complex Behavior Simulation**
  - Add environmental feedback loops
  - Implement basic avoidance learning

## Author

This project is developed by [@Ibrahim](https://github.com/iBz-04)

