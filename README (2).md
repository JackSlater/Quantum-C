# 🧠 Quantum-C: A High-Performance Quantum Computing Implementation in C++

## 🚀 Overview

**Quantum-C** is a C++-based quantum computing framework designed to
simulate and benchmark quantum operations such as **Grover's
algorithm**, **quantum echo experiments**, and **qubit register
manipulations**.\
It was developed to compare **runtime efficiency** and **scalability**
between classical C++ simulation and **Python-based Qiskit runtimes**
(IBM Cloud backends such as *ibm_brisbane*).

## 🧩 Key Features

-   **Pure C++17** implementation with modular architecture.\
-   **Qubit Simulation Engine** supporting state vectors and measurement
    collapse.\
-   **Grover's Search** algorithm runtime comparison vs Python
    implementation.\
-   **Quantum Echo** experiment simulation with tunable circuit depth.\
-   **Benchmarking Suite** for evaluating runtime growth with qubit
    count.\
-   **Cross-language Validation** against Python and IBM Qiskit
    results.\
-   **Thread-safe timing and data logging utilities.**

## 📊 Research Findings

  ----------------------------------------------------------------------------------
   Qubits     Python Runtime (s)     C++ Runtime (s)      Speedup   Notes
  --------- ---------------------- -------------------- ----------- ----------------
      2             0.006                 0.003            \~2×     Initialization
                                                                    overhead smaller
                                                                    in C++

      3             0.003                 0.002           \~1.5×    Near-linear
                                                                    scaling

      5             0.004                 0.003           \~1.3×    Maintains stable
                                                                    latency

      8       22.6 (IBM runtime)        0.05 (C++)        \~450×    Significant
                                                                    advantage in
                                                                    local simulation
  ----------------------------------------------------------------------------------

**Result:**\
C++ outperforms Python by an order of magnitude for local simulations,
especially in qubit echo experiments.\
However, IBM hardware runs introduce physical constraints that pure
simulation cannot replicate --- showing the trade-off between
**simulation speed** and **real quantum noise fidelity**.

## ⚙️ Usage

### 🔧 Build

``` bash
g++ -std=c++17 -O3 -o quantum quantum.cpp
```

### ▶️ Run Example

``` bash
./quantum --mode grover --qubits 5 --iterations 4
```

### 🧪 Options

  ------------------------------------------------------------------------
  Flag           Description                        Example
  -------------- ---------------------------------- ----------------------
  `--mode`       Choose experiment: `grover`,       `--mode echo`
                 `echo`, `benchmark`                

  `--qubits`     Set number of qubits               `--qubits 5`

  `--depth`      Circuit depth (for echo)           `--depth 8`

  `--shots`      Number of measurement samples      `--shots 1024`
  ------------------------------------------------------------------------

## 🧠 Architecture

    src/
     ├── core/
     │   ├── qubit.hpp        # Qubit state and operators
     │   ├── gate.hpp         # Quantum gate matrices
     │   └── utils.hpp        # Timing + random number helpers
     ├── experiments/
     │   ├── grover.cpp
     │   ├── echo.cpp
     │   └── benchmark.cpp
     └── main.cpp

## 🔬 Future Work

-   Integration with **Qiskit Runtime Service** via REST bridge.\
-   GPU acceleration for multi-qubit simulation using CUDA/OpenCL.\
-   Visualization of state vectors and gate operations in browser.

## 🧾 Citation

If using results or methodology from this work, please cite:

> Slater, J. (2025). *Quantum-C: Efficient Quantum Simulation and
> Runtime Analysis in C++ Compared to IBM Qiskit*. University of
> Colorado Boulder.
