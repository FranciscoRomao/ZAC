import time
import math
import rustworkx as rx

class Scheduler_mixin:
    def scheduling(self):
        """
        solve gate scheduling problem for all-commutable gate cases by graph coloring algorithm
        """
        t_s = time.time()
        self.gate_scheduling = []
        if self.has_dependency:
            if self.scheduling_strategy == "asap":
                result_scheduling = self.asap()
            else:
                result_scheduling = [[i] for i in range(len(self.g_q))]
        else:
            result_scheduling = self.graph_coloring()
        
        # TODO: handle the case that the number of gates in one layer exceed the capacity of rydberg zone
        max_gate_num = 0
        for zone in self.architecture.entanglement_zone:
            slm = self.architecture.dict_SLM[zone[0]]
            max_gate_num += (slm.n_r * slm.n_c)
            # break
        
        self.gate_scheduling_idx = []
        self.barriers_per_scheduled_layer = []
        barriers_by_time = {}
        for barrier_index, barrier_time in self.barrier_times.items():
            barriers_by_time.setdefault(barrier_time, []).append(barrier_index)

        for logical_layer, gates in enumerate(result_scheduling):
            barrier_indices = barriers_by_time.get(logical_layer, [])
            if len(gates) < max_gate_num:
                self.gate_scheduling_idx.append(gates)
                self.barriers_per_scheduled_layer.append(barrier_indices)
            else:
                num_layer = math.ceil(len(gates) / max_gate_num)
                gates_per_layer = math.ceil(len(gates) / num_layer)
                for chunk_start in range(0, len(gates), gates_per_layer):
                    self.gate_scheduling_idx.append(
                        gates[chunk_start: chunk_start + gates_per_layer]
                    )
                    self.barriers_per_scheduled_layer.append(
                        barrier_indices if chunk_start == 0 else []
                    )

        self.gate_scheduling = []
        self.gate_1q_scheduling = []
        self.gate_1q_barrier_qubits = []

        for gates, barrier_indices in zip(
            self.gate_scheduling_idx, self.barriers_per_scheduled_layer
        ):
            tmp = [self.g_q[i] for i in gates]
            self.gate_scheduling.append(tmp)
            self.gate_1q_scheduling.append([])
            self.gate_1q_barrier_qubits.append(set())
            for gate_idx in gates:
                if gate_idx in self.dict_g_1q_parent:
                    for gate_1q in self.dict_g_1q_parent[gate_idx]:
                        self.gate_1q_scheduling[-1].append(gate_1q)
            for barrier_index in barrier_indices:
                self.gate_1q_scheduling[-1].extend(
                    self.dict_g_1q_parent.get(barrier_index, [])
                )
                self.gate_1q_barrier_qubits[-1].update(
                    self.barrier_qubits[barrier_index]
                )
        
        self.runtime_analysis["scheduling"] = time.time()- t_s
        
        print("[INFO]               Time for scheduling: {}s".format(self.runtime_analysis["scheduling"]))

    def asap(self):
        # as soon as possible algorithm for self.g_q
        gate_scheduling = []
        list_qubit_time = [0 for i in range(self.n_q)]
        self.barrier_times = {}
        for i, gate in enumerate(self.g_q):
            if gate == [-1, -1]:
                # Old callers can still provide a bare marker, which retains
                # the historical full-circuit-barrier behavior.
                barrier_qubits = getattr(self, "barrier_qubits", {}).get(
                    i, set(range(self.n_q))
                )
                max_t = max((list_qubit_time[q] for q in barrier_qubits), default=0)
                for q in barrier_qubits:
                    list_qubit_time[q] = max_t
                self.barrier_times[i] = max_t

                barrier_1q_gates = self.dict_g_1q_parent.get(i, [])
                is_full_barrier = barrier_qubits == set(range(self.n_q))
                if is_full_barrier or barrier_1q_gates:
                    while len(gate_scheduling) <= max_t:
                        gate_scheduling.append([])
                    for _, qubit in barrier_1q_gates:
                        list_qubit_time[qubit] = max_t + 1

                # Only an all-qubit barrier is an execution/placement boundary.
                if is_full_barrier:
                    self.barrier_layers[max_t] = (
                        self.barrier_layers.get(max_t, 0) + 1
                    )
                continue

            tq0 = list_qubit_time[gate[0]]
            tq1 = list_qubit_time[gate[1]]
            tg = max(tq0, tq1)
            if tg >= len(gate_scheduling):
                gate_scheduling.append([])
            gate_scheduling[tg].append(i)

            tg += 1
            list_qubit_time[gate[0]] = tg
            list_qubit_time[gate[1]] = tg
        return gate_scheduling
    
    def graph_coloring(self):
        graph = rx.PyGraph()
        graph.add_nodes_from(list(range(self.n_q)))
        for edge in self.g_q:
            if edge == [-1, -1]:
                continue
            graph.add_edge(edge[0], edge[1], edge)
        edge_colors = rx.graph_misra_gries_edge_color(graph)
        max_color = 0
        for i in edge_colors:
            max_color = max(max_color, edge_colors[i])
        gate_scheduling = [[] for i in range(max_color + 1)]
        for i in range(len(self.g_q)):
            gate_scheduling[edge_colors[i]].append(i)
        return gate_scheduling
