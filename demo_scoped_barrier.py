"""Compare scheduling with and without a scoped barrier."""

import os
import tempfile
import json
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

from zac.ds.architecture import Architecture
from zac.zac import ZAC


def compile_circuit(circuit):
    """Compile a circuit through ZAC's normal placement and routing pipeline."""

    with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as qasm_file:
        qasm_path = qasm_file.name

    try:
        compiler = ZAC()
        compiler.resyn = False
        compiler.parse_setting(
            {
                "name": "scoped_barrier_demo",
                "dependency": True,
                "routing_strategy": "maximalis",
                "dynamic_placement": True,
                "reuse": True,
                "use_window": True,
                "window_size": 20,
                "use_verifier": False,
            }
        )
        architecture_path = Path("hardware_spec/half_architecture.json")
        architecture = Architecture(json.loads(architecture_path.read_text()))
        architecture.preprocessing()
        compiler.set_architecture(architecture)
        compiler.set_architecture_spec_path(str(architecture_path))
        qasm2.dump(circuit, qasm_path)
        compiler.set_program(qasm_path)
        result = compiler.solve(save_file=False)
    finally:
        os.unlink(qasm_path)
    return result


def print_zac_output(result):
    """Print ZAC output as NDJSON, with one generated instruction per line."""
    metadata = {
        "name": result["name"],
        "architecture_spec_path": result["architecture_spec_path"],
        "runtime": result["runtime"],
        "execution_segment_instruction_counts": [
            len(segment) for segment in result["execution_segments"]
        ],
    }
    print(json.dumps({"metadata": metadata}))
    for instruction in result["instructions"]:
        print(json.dumps(instruction, separators=(",", ":")))


def main():
    # The first two gates make qubit 0 available only at layer 2.
    # A scoped barrier synchronizes qubit 1 to that time.  The gate on 1–4
    # must therefore wait, whereas the gate on unrelated qubits 2–5 does not.
    with_scoped_barrier = QuantumCircuit(6)
    with_scoped_barrier.cx(0, 2)
    with_scoped_barrier.cx(0, 3)
    with_scoped_barrier.h(0)
    with_scoped_barrier.barrier(0, 1)
    with_scoped_barrier.h(1)
    with_scoped_barrier.cx(1, 4)
    with_scoped_barrier.cx(2, 5)

    print("Circuit with scoped barrier on qubits 0 and 1:")
    print(with_scoped_barrier.draw(output="text"))
    result = compile_circuit(with_scoped_barrier)
    print("\nFull ZAC output (one JSON object per line):")
    print_zac_output(result)


if __name__ == "__main__":
    main()
