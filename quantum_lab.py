# quantum_lab.py — cleaned for C++ embedding (no top-level side effects)

import os
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

# Avoid GUI popups from matplotlib if any plotting happens internally
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------
#  Qiskit imports
# ---------------------------
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Aer (local sim) for fast timing runs
from qiskit_aer.primitives import SamplerV2 as AerSampler

# IBM Runtime primitives
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)


# ---------------------------
#  IBM Runtime helpers
# ---------------------------
def _get_service() -> QiskitRuntimeService:
    """
    Return a QiskitRuntimeService using env vars:
      - IBM_QUANTUM_TOKEN (recommended)
      - IBM_QUANTUM_INSTANCE (optional)
    If you've previously saved an account locally, it will reuse it.
    """
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE", "")

    if token:
        try:
            # Save once; future calls reuse the stored account
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=token,
                overwrite=False,
            )
        except Exception:
            # Already saved or not needed
            pass

    if instance:
        return QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)
    return QiskitRuntimeService(channel="ibm_quantum_platform")


def _select_backend(min_qubits: int = 1, name: Optional[str] = None):
    """
    Choose an operational backend with at least min_qubits.
    If 'name' is provided, return that backend.
    """
    svc = _get_service()
    if name:
        return svc.backend(name)

    # Filter for operational backends with enough qubits
    candidates = []
    for b in svc.backends():
        try:
            if getattr(b, "status")().operational and getattr(b, "num_qubits", 0) >= min_qubits:
                candidates.append(b)
        except Exception:
            # Some backends may not respond well to status(); ignore
            pass

    if candidates:
        # Heuristic: pick the first; you can sort by queue length if desired
        return candidates[0]

    # Fallback to least_busy(), but ensure it meets qubit requirement
    lb = svc.least_busy()
    if getattr(lb, "num_qubits", 0) >= min_qubits:
        return lb
    raise RuntimeError(
        f"No available backend meets min_qubits={min_qubits}. "
        f"Least-busy '{getattr(lb,'name',str(lb))}' has {getattr(lb,'num_qubits',0)}."
    )


# ---------------------------
#  Grover helpers (lightweight)
# ---------------------------
def _phase_oracle(bitstring: str) -> QuantumCircuit:
    n = len(bitstring)
    qc = QuantumCircuit(n, name="Oracle")
    # Map target to all-ones with X on zeros (reverse to align with Qiskit's qubit order)
    for i, b in enumerate(reversed(bitstring)):
        if b == "0":
            qc.x(i)
    # Phase flip on |11..1| via H - MCX - H on qubit 0
    qc.h(0)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(1, n)), 0)
    qc.h(0)
    # Uncompute the Xs
    for i, b in enumerate(reversed(bitstring)):
        if b == "0":
            qc.x(i)
    return qc


def _diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(0)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(1, n)), 0)
    qc.h(0)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def _grover_circuit(bitstring: str, iters: int) -> QuantumCircuit:
    n = len(bitstring)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    O = _phase_oracle(bitstring)
    D = _diffuser(n)
    for _ in range(iters):
        qc.compose(O, inplace=True)
        qc.compose(D, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def _optimal_iters(n: int, marked_count: int = 1) -> int:
    return max(1, int(math.floor((math.pi / 4) * math.sqrt(2**n / marked_count))))


def measure_runtime_vs_size(n_values: List[int], shots: int = 2000) -> List[Tuple[int, float, int, float]]:
    """
    Uses Aer Sampler to time local Grover runs.
    Returns list of (n, sqrtN, k, wall_seconds).
    """
    sampler = AerSampler()
    out: List[Tuple[int, float, int, float]] = []
    for n in n_values:
        target = "1" * n
        k = _optimal_iters(n)
        qc = _grover_circuit(target, k)
        t0 = time.perf_counter()
        sampler.run([qc], shots=shots).result()
        t1 = time.perf_counter()
        out.append((n, math.sqrt(2**n), k, t1 - t0))
    return out


# ---------------------------
#  Echo experiment (estimator)
# ---------------------------
@dataclass
class EchoConfig:
    n_qubits: int = 12
    depth: int = 16
    pert_qubit: int = 0
    basis: str = "x"             # "x" | "y" | "z"
    shots: int = 4096
    seed: int = 42
    optimization_level: int = 1


def _build_U(n: int, d: int, seed: int) -> QuantumCircuit:
    """
    A simple random layered unitary U to create scrambling.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for l in range(d):
        # Single-qubit random rotations
        for q in range(n):
            qc.rx(float(rng.uniform(0, 2 * np.pi)), q)
            qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
        # Entangling pattern (even/odd layers)
        if l % 2 == 0:
            for i in range(0, n - 1, 2):
                qc.cx(i, i + 1)
        else:
            for i in range(1, n - 1, 2):
                qc.cx(i, i + 1)
    return qc


def make_echo_circuit(cfg: EchoConfig) -> QuantumCircuit:
    U = _build_U(cfg.n_qubits, cfg.depth, cfg.seed)
    qc = QuantumCircuit(cfg.n_qubits, name="echo")
    qc.compose(U, inplace=True)
    b = cfg.basis.lower()
    if b == "x":
        qc.x(cfg.pert_qubit)
    elif b == "y":
        qc.y(cfg.pert_qubit)
    else:
        qc.z(cfg.pert_qubit)
    qc.compose(U.inverse(), inplace=True)
    return qc


def _fullwidth_z(idx: int, width: int) -> SparsePauliOp:
    s = ["I"] * width
    s[idx] = "Z"
    return SparsePauliOp.from_list([("".join(s), 1.0)])


def _logical_to_physical_index(tqc: QuantumCircuit, q: int) -> int:
    """
    Try to map the logical qubit 'q' to its physical index after transpilation.
    Falls back to 'q' if layout data isn't present.
    """
    lay = getattr(tqc, "layout", None)
    if lay is not None:
        # Qiskit 1.x layouts can differ; check a couple of known attributes
        fim = getattr(lay, "final_index_map", None)
        if isinstance(fim, dict) and q in fim:
            return int(fim[q])
        fl = getattr(lay, "final_layout", None)
        if fl is not None:
            try:
                virt = list(fl.get_virtual_bits().keys())[q]
                return int(fl[virt].index)
            except Exception:
                pass
    return q


def run_echo_estimator(cfg: EchoConfig, backend_name: Optional[str] = None) -> Dict[str, float | int | str]:
    """
    Submits one echo circuit to IBM via Estimator and returns a small dict.
    Requires that IBM_QUANTUM_TOKEN is set (or an account has been saved previously).
    """
    service = _get_service()
    backend = _select_backend(min_qubits=cfg.n_qubits, name=backend_name)

    qc = make_echo_circuit(cfg)

    # Transpile with a preset pass manager for the chosen backend
    pm = generate_preset_pass_manager(
        optimization_level=cfg.optimization_level, backend=backend
    )
    tqc = pm.run(qc)

    phys = _logical_to_physical_index(tqc, cfg.pert_qubit)
    obs = _fullwidth_z(phys, backend.num_qubits)

    est = Estimator(mode=backend, options={"default_shots": cfg.shots})

    t0 = time.perf_counter()
    res = est.run([(tqc, obs, [])]).result()[0]
    wall = time.perf_counter() - t0

    # Extract scalar ev/std
    ev = float(np.asarray(res.data.evs).reshape(-1)[0])
    std = float(np.asarray(res.data.stds).reshape(-1)[0])

    return {
        "backend": getattr(backend, "name", str(backend)),
        "phys_index": int(phys),
        "ev": ev,
        "std": std,
        "wall_seconds": float(wall),
        "shots": int(cfg.shots),
        "depth": int(cfg.depth),
        "n_qubits": int(cfg.n_qubits),
        "basis": cfg.basis,
    }


# ---------------------------
#  C++-friendly wrappers
# ---------------------------
def py_measure_runtime_vs_size(n_min: int = 2, n_max: int = 8, shots: int = 2000):
    """
    Returns 4 parallel lists so C++ can cast cleanly:
      ns, sqrtNs, iters, secs
    """
    ns = list(range(int(n_min), int(n_max) + 1))
    data = measure_runtime_vs_size(ns, shots=int(shots))
    return (
        [int(d[0]) for d in data],
        [float(d[1]) for d in data],
        [int(d[2]) for d in data],
        [float(d[3]) for d in data],
    )


def py_run_echo(
    depth: int = 10,
    shots: int = 4096,
    n_qubits: int = 12,
    basis: str = "x",
    seed: int = 42,
    backend_name: Optional[str] = None,
):
    """
    Run a single echo experiment via IBM Runtime (Estimator) and return a small,
    JSON-serializable dict for easy use from C++.

    Requires IBM_QUANTUM_TOKEN (and optionally IBM_QUANTUM_INSTANCE) to be set
    in the environment or an account saved previously via QiskitRuntimeService.
    """
    try:
        cfg = EchoConfig(
            n_qubits=int(n_qubits),
            depth=int(depth),
            pert_qubit=0,
            basis=str(basis),
            shots=int(shots),
            seed=int(seed),
            optimization_level=1,
        )
        return run_echo_estimator(cfg, backend_name=backend_name)
    except Exception as e:
        raise RuntimeError(
            "py_run_echo failed. Ensure IBM_QUANTUM_TOKEN is set (and optionally "
            "IBM_QUANTUM_INSTANCE), or that an IBM account is saved on this machine. "
            f"Original error: {e}"
        )


# ---------------------------
#  Optional: local main for ad-hoc testing
# ---------------------------
if __name__ == "__main__":
    # Example manual runs (won't execute when imported by C++)
    print("Demo: timing Grover locally...")
    ns, sqrtNs, iters, secs = py_measure_runtime_vs_size(2, 5, 512)
    for i in range(len(ns)):
        print(f"n={ns[i]} sqrtN={sqrtNs[i]:.2f} k={iters[i]} secs={secs[i]:.4f}")

    # Echo demo requires IBM credentials; uncomment if set
    # print("Demo: running echo on IBM...")
    # print(py_run_echo(depth=8, n_qubits=5, shots=1024, basis="x"))
