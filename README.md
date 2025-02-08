# br-AI-n

This project is the beginning of a new type of artificial intelligence—one that is not based on statistical learning but on biological principles. Instead of using artificial neural networks trained on data, my goal is to build AI that functions like a human brain, neuron by neuron, system by system, everything hard coded. The approach is questionable but hey, let's try.

The current phase starts small, with the simplest form of neural processing: the reflex arc. This is the foundation of all higher intelligence, as even complex thought processes are built upon basic neural circuits like these.

<img src="https://res.cloudinary.com/diekemzs9/image/upload/v1739055438/Neuron_ltkwqy.png" alt="Neuron" width="800"/>

At its core, our system consists of:
- **Biologically accurate neurons** that have real-time signal processing, thresholds, decay rates, and refractory periods.
- **A structured nervous system** that processes signals in a way similar to how a human nervous system would.
- **A core task (escaping danger)** as the first “instinct” of our artificial brain, before we add more cognitive functions.

The end goal is not just to create AI that “predicts” based on patterns but AI that thinks, adapts, and functions like a living brain—not through pre-programmed knowledge but through organic learning and decision-making.

## Overview

In this initial phase, we simulate a basic reflex arc composed of three interconnected neurons:
- **Sensor Neuron:** Receives a danger signal as input.
- **Interneuron:** Processes the sensor neuron's signal.
- **Motor Neuron:** Triggers an escape response when activated.

Each neuron follows an integrate-and-fire model that:
- **Integrates** inputs over discrete time steps.
- Applies a **decay** to mimic the natural loss of membrane potential.
- **Fires** when the accumulated potential exceeds a set threshold.
- Resets and enters a **refractory period** post firing.

## Current Code

The present implementation includes the following key components:

- **brain.py**  
  Contains the main simulation logic:
  - `Neuron` class – Implements the basic relational model of a neuron.
  - `ReflexArc` class – Orchestrates interactions between the sensor, interneuron, and motor neuron.

- **visualize.py**  
  Leverages Matplotlib to graphically represent the firing events of individual neurons over time, offering insights into signal propagation within the reflex arc.

- **test_brain.py**  
  Provides unit tests to verify behaviors such as firing thresholds, refractory periods, and correct cascade signals among neurons.

## Requirements

- Python 3.x (tested on Python 3.12)

## Future Enhancements

Looking ahead, we plan to evolve this project by:
- **Enhancing Neuron Models:**  
  Introducing adaptive thresholds and variable decay rates for more realistic dynamics.
- **Implementing Synaptic Plasticity:**  
  Simulating learning by adjusting connection weights based on activity.
- **Expanding Neural Circuits:**  
  Building more elaborate networks to emulate higher-level cognitive functions.
- **Advanced Visualization:**  
  Developing sophisticated visual tools to monitor network dynamics and neuron interactions.


## Author

This project is developed by [@Ibrahim](https://github.com/iBz-04)

