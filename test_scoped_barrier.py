import os
import tempfile
import unittest
from types import SimpleNamespace

from qiskit import QuantumCircuit, qasm2

from zac.zac import ZAC


class ScopedBarrierTest(unittest.TestCase):
    def parse(self, circuit):
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as qasm_file:
            qasm_path = qasm_file.name
        try:
            qasm2.dump(circuit, qasm_path)
            compiler = ZAC()
            compiler.resyn = False
            compiler.set_program(qasm_path)
            return compiler
        finally:
            os.unlink(qasm_path)

    def schedule(self, compiler):
        compiler.architecture = SimpleNamespace(
            entanglement_zone=[(0,)],
            dict_SLM={0: SimpleNamespace(n_r=10, n_c=10)},
        )
        compiler.scheduling()

    def test_scoped_barrier_fences_only_its_qubits(self):
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.barrier(0, 1)
        circuit.cx(2, 3)
        circuit.cx(0, 2)

        compiler = self.parse(circuit)

        self.assertEqual(compiler.barrier_qubits, {1: {0, 1}})
        self.assertEqual(compiler.asap(), [[0, 2], [3]])
        self.assertEqual(compiler.barrier_layers, set())

    def test_full_barrier_remains_an_execution_boundary(self):
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.cx(2, 3)

        compiler = self.parse(circuit)

        self.assertEqual(compiler.barrier_qubits, {1: {0, 1, 2, 3}})
        self.assertEqual(compiler.asap(), [[0], [2]])
        self.assertEqual(compiler.barrier_layers, {1})

    def test_one_qubit_gate_after_scoped_barrier_waits_for_the_barrier(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 3)
        circuit.cx(0, 4)
        circuit.cx(1, 2)
        circuit.barrier(0, 1)
        circuit.h(1)
        circuit.cx(1, 2)

        compiler = self.parse(circuit)
        self.schedule(compiler)

        self.assertEqual(compiler.gate_scheduling_idx, [[0, 2], [1], [], [4]])
        self.assertEqual(
            compiler.gate_1q_scheduling,
            [[], [], [("h", 1)], []],
        )


if __name__ == "__main__":
    unittest.main()
