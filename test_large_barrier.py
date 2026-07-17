import json
from qiskit import QuantumCircuit, transpile, qasm2
from mqt.bench import BenchmarkLevel, get_benchmark
from zac.zac import ZAC
from zac.ds.architecture import Architecture

print("--- Fetching QFT benchmark from mqt.bench ---")
circuit = get_benchmark("qft", BenchmarkLevel.INDEP, circuit_size=20)
circuit.remove_final_measurements()

# 1. No barrier circuit
qc_no_barrier = transpile(circuit, basis_gates=['cz', 'id', 'u2', 'u1', 'u3'], optimization_level=3)
qasm2.dump(qc_no_barrier, "no_barrier.qasm")

# 2. Barrier circuit
new_qc = QuantumCircuit(circuit.num_qubits)
total_ops = len(circuit.data)
p1 = total_ops // 3
p2 = 2 * total_ops // 3

for i, inst in enumerate(circuit.data):
    if i == p1 or i == p2:
        new_qc.barrier()
    if inst.operation.name != "measure":
        new_qc.append(inst)

qc_barrier = transpile(new_qc, basis_gates=['cz', 'id', 'u2', 'u1', 'u3'], optimization_level=3)
qasm2.dump(qc_barrier, "with_barrier.qasm")

def get_stages(qasm_path):
    compiler = ZAC()
    compiler.parse_setting({
        "name": "test",
        "dependency": True,
        "routing_strategy": "maximalis",
        "dynamic_placement": True,
        "reuse": True,
        "use_window": True,
        "window_size": 20
    })
    with open("hardware_spec/half_architecture.json", 'r') as f:
        spec = json.load(f)
    arch = Architecture(spec)
    arch.preprocessing()
    compiler.set_architecture(arch)
    compiler.set_program(qasm_path)
    
    # We only need to run scheduling to see the effect of the barrier
    compiler.scheduling()
    return len(compiler.gate_scheduling)

print("\n--- Running ZAC Scheduler ---")
stages_no_barrier = get_stages("no_barrier.qasm")
stages_with_barrier = get_stages("with_barrier.qasm")

print(f"\nStages WITHOUT barrier: {stages_no_barrier}")
print(f"Stages WITH barriers:   {stages_with_barrier}")
print(f"Depth increased by:     {stages_with_barrier - stages_no_barrier} stages")
