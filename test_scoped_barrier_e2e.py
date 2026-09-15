"""End-to-end barrier ordering checks against ZAC's emitted instructions."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

from zac.ds.architecture import Architecture
from zac.zac import ZAC


class ScopedBarrierEndToEndTest(unittest.TestCase):
    def compile(self, circuit, name):
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as qasm_file:
            qasm_path = qasm_file.name

        try:
            qasm2.dump(circuit, qasm_path)
            compiler = ZAC()
            compiler.resyn = False
            compiler.parse_setting(
                {
                    "name": name,
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
            compiler.set_program(qasm_path)
            return compiler.solve(save_file=False)
        finally:
            os.unlink(qasm_path)

    def one_qubit_instruction(self, result, qubit):
        matches = [
            instruction
            for instruction in result["instructions"]
            if instruction["type"] == "1qGate"
            and any(gate["name"] == "h" and gate["q"] == qubit for gate in instruction["gates"])
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def two_qubit_instruction(self, result, qubits):
        expected_qubits = set(qubits)
        matches = [
            instruction
            for instruction in result["instructions"]
            if instruction["type"] == "rydberg"
            and any({gate["q0"], gate["q1"]} == expected_qubits for gate in instruction["gates"])
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def assert_after(self, later, earlier):
        self.assertGreaterEqual(later["begin_time"], earlier["end_time"])

    def test_five_qubit_scoped_barrier_orders_a_one_qubit_gate(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.h(0)
        circuit.barrier(0, 1)
        circuit.h(1)
        circuit.cx(1, 4)

        result = self.compile(circuit, "five_qubit_scoped_barrier")

        h_before = self.one_qubit_instruction(result, 0)
        h_after = self.one_qubit_instruction(result, 1)
        later_two_qubit_gate = self.two_qubit_instruction(result, (1, 4))
        self.assert_after(h_after, h_before)
        self.assert_after(later_two_qubit_gate, h_after)

    def test_six_qubit_scoped_barrier_allows_an_unrelated_gate(self):
        circuit = QuantumCircuit(6)
        circuit.cx(0, 3)
        circuit.cx(0, 4)
        circuit.h(0)
        circuit.barrier(0, 1)
        circuit.h(1)
        circuit.cx(2, 5)
        circuit.cx(1, 4)

        result = self.compile(circuit, "six_qubit_scoped_barrier")

        h_before = self.one_qubit_instruction(result, 0)
        h_after = self.one_qubit_instruction(result, 1)
        later_scoped_gate = self.two_qubit_instruction(result, (1, 4))
        self.assert_after(h_after, h_before)
        self.assert_after(later_scoped_gate, h_after)
        self.assertEqual(len(result["execution_segments"]), 1)

    def test_seven_qubit_two_target_barrier_orders_a_new_target_gate(self):
        circuit = QuantumCircuit(7)
        circuit.cx(1, 4)
        circuit.cx(2, 5)
        circuit.h(1)
        circuit.barrier(1, 2)
        circuit.h(2)
        circuit.cx(2, 6)
        circuit.cx(0, 3)

        result = self.compile(circuit, "seven_qubit_two_target_barrier")

        h_before = self.one_qubit_instruction(result, 1)
        h_after = self.one_qubit_instruction(result, 2)
        later_scoped_gate = self.two_qubit_instruction(result, (2, 6))
        self.assert_after(h_after, h_before)
        self.assert_after(later_scoped_gate, h_after)

    def test_eight_qubit_consecutive_scoped_barriers_preserve_order(self):
        circuit = QuantumCircuit(8)
        circuit.cx(0, 4)
        circuit.cx(0, 5)
        circuit.h(0)
        circuit.barrier(0, 1)
        circuit.h(1)
        circuit.cx(1, 6)
        circuit.barrier(1, 2)
        circuit.h(2)
        circuit.cx(2, 7)

        result = self.compile(circuit, "eight_qubit_consecutive_barriers")

        h_zero = self.one_qubit_instruction(result, 0)
        h_one = self.one_qubit_instruction(result, 1)
        h_two = self.one_qubit_instruction(result, 2)
        between_barriers = self.two_qubit_instruction(result, (1, 6))
        final_scoped_gate = self.two_qubit_instruction(result, (2, 7))
        self.assert_after(h_one, h_zero)
        self.assert_after(h_two, between_barriers)
        self.assert_after(final_scoped_gate, h_two)

    def test_nine_qubit_three_target_barrier_waits_for_all_targets(self):
        circuit = QuantumCircuit(9)
        circuit.cx(0, 3)
        circuit.cx(1, 4)
        circuit.cx(2, 5)
        circuit.h(0)
        circuit.barrier(0, 1, 2)
        circuit.h(2)
        circuit.cx(2, 6)
        circuit.cx(7, 8)

        result = self.compile(circuit, "nine_qubit_three_target_barrier")

        h_before = self.one_qubit_instruction(result, 0)
        h_after = self.one_qubit_instruction(result, 2)
        later_scoped_gate = self.two_qubit_instruction(result, (2, 6))
        self.assert_after(h_after, h_before)
        self.assert_after(later_scoped_gate, h_after)

    def test_ten_qubit_full_barrier_keeps_execution_segments(self):
        circuit = QuantumCircuit(10)
        circuit.cx(0, 5)
        circuit.cx(0, 6)
        circuit.h(0)
        circuit.cx(1, 7)
        circuit.barrier()
        circuit.h(9)
        circuit.cx(8, 9)
        circuit.cx(2, 3)

        result = self.compile(circuit, "ten_qubit_full_barrier")

        h_before = self.one_qubit_instruction(result, 0)
        h_after = self.one_qubit_instruction(result, 9)
        later_full_barrier_gate = self.two_qubit_instruction(result, (8, 9))
        self.assertEqual(len(result["execution_segments"]), 2)
        self.assert_after(h_after, h_before)
        self.assert_after(later_full_barrier_gate, h_after)

    def test_consecutive_full_barriers_preserve_empty_segment_multiplicity(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.barrier()
        circuit.barrier()
        circuit.cx(2, 3)
        circuit.h(4)

        result = self.compile(circuit, "consecutive_full_barriers")

        self.assertEqual(len(result["execution_segments"]), 3)

    def test_trailing_full_barrier_creates_an_empty_final_segment(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(2, 3)
        circuit.h(2)
        circuit.barrier()

        result = self.compile(circuit, "trailing_full_barrier")

        self.assertEqual(len(result["execution_segments"]), 2)
        self.assertEqual(result["execution_segments"][-1], [])

    def test_consecutive_trailing_full_barriers_preserve_both_empty_segments(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(2, 3)
        circuit.h(2)
        circuit.barrier()
        circuit.barrier()

        result = self.compile(circuit, "consecutive_trailing_full_barriers")

        self.assertEqual(len(result["execution_segments"]), 3)
        self.assertEqual(result["execution_segments"][-2:], [[], []])


if __name__ == "__main__":
    unittest.main()
