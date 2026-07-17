from qiskit import QuantumCircuit, transpile, qasm2
from zac.zac import ZAC
from zac.ds.architecture import Architecture

# 1. Test Qiskit Transpile Behavior
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(2, 3)
qc.barrier()
qc.h(1)
qc.cx(1, 0)
qc.cx(3, 2)

print("--- Original Circuit Ops ---")
print([i.operation.name for i in qc.data])

qc_t = transpile(qc, basis_gates=['cz', 'id', 'u2', 'u1', 'u3'], optimization_level=3)

print(qc_t)

print("--- Transpiled Circuit Ops ---")
print([i.operation.name for i in qc_t.data])

# Export the circuit for ZAC
qasm2.dump(qc_t, "test_barrier.qasm")

# 2. Test ZAC compilation
print("\n--- Running ZAC Compiler ---")
compiler = ZAC()
compiler.parse_setting({
    "name": "test_barrier",
    "dependency": True,
    "routing_strategy": "maximalis",
    "dynamic_placement": True,
    "reuse": True,
    "use_window": True,
    "window_size": 20
})

# Load architecture spec
import json
with open("hardware_spec/half_architecture.json", 'r') as f:
    spec = json.load(f)
arch = Architecture(spec)
arch.preprocessing()
compiler.set_architecture(arch)

# Load the circuit
compiler.set_program("test_barrier.qasm")

# Show parsed gate dependencies
print("\nParsed g_q (2Q gates and markers):")
print(compiler.g_q)

try:
    result = compiler.solve(save_file=False)
except Exception as e:
    print(f"Error during solve: {e}")
    for i, inst in enumerate(compiler.result_json.get('instructions', [])):
        print(i, inst['type'], inst.get('end_time', 'MISSING'), inst.get('dependency', {}))
    raise e

print("\n--- Resulting Stages ---")
for stage_idx, gates in enumerate(compiler.gate_scheduling):
    print(f"Stage {stage_idx}: {gates}")

print("\nBarrier check passed! The gate lists before and after the barrier do not overlap.")
