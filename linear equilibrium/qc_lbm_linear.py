import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
import qiskit_aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.circuit.library import HGate, XGate
from enum import Enum
from math import ceil

kCs = 1. / np.sqrt(3.)
kEps = 1e-8

class IqDir(Enum):
    x_neg = 0
    x_pos = 1
    y_neg = 2
    y_pos = 3
    z_neg = 4
    z_pos = 5
class BoundaryType(Enum):
    x_min = 0
    x_max = 1
    y_min = 2
    y_max = 3
    z_min = 4
    z_max = 5

class QcLbm:
    # LBM related
    dim = None
    nq = None
    cx = None
    cy = None
    cz = None
    wi = None
    nx = 1
    ny = 1
    nz = 1
    omega = None

    # quantum computed
    num_qubit_f = None
    num_qubit_ancilla = 1
    num_max_variables = None

    def FlattenInitialField(self, initial_field: np.ndarray) -> np.ndarray:
        pass
    @staticmethod
    def GetIqBasedOnDirection(dir: IqDir) -> np.ndarray:
        pass
    def ReplacePeriodicVariablesForCollisionMatrix(self, boundary_type: BoundaryType,\
        replacement_matrix: np.ndarray, collision_matrix: np.ndarray):
        pass
    def SetBounceBackBoundaryForCollisionMatrix(self, boundary_type: BoundaryType, u_0: float,\
        boundary_matrix : np.ndarray, collision_matrix: np.ndarray, initial_field_flattened: np.ndarray):
        pass
    def ComputeCollisionMatrix(self, omega: float) -> np.ndarray:
        pass
    def AddForceTermToCollisionMatrix(self, force: np.ndarray, collision_matrix: np.ndarray, initial_field_flattened: np.ndarray):
        pass
    def GetLatticeVelocities(self) -> np.ndarray:
        pass

    def InitialEncoding(self, initial_field_flattened: np.ndarray) -> tuple[float, QuantumCircuit]:
        normalize_scale = np.linalg.norm(initial_field_flattened)
        initial_field_normalized = initial_field_flattened / normalize_scale

        q_coordinates = []
        if self.dim == 3:
            q_coordinates.append(QuantumRegister(ceil(np.log2(self.nz)), 'z'))
        if self.dim >= 2:
            q_coordinates.append(QuantumRegister(ceil(np.log2(self.ny)), 'y'))
        q_coordinates.append(QuantumRegister(ceil(np.log2(self.nx)), 'x'))

        q_f = QuantumRegister(self.num_qubit_f, 'f')
        q_ancilla = QuantumRegister(self.num_qubit_ancilla, 'a')

        qc = QuantumCircuit(q_f, *q_coordinates, q_ancilla)

        qc.reset(q_ancilla)
        coordinate_qubits = [q for reg in q_coordinates for q in reg]
        initial_qubits = list(q_f) + coordinate_qubits
        qc.initialize(initial_field_normalized, initial_qubits)
        
        qc.barrier()

        return normalize_scale, qc
    
    @staticmethod
    def PositiveShift(reg_i_dim: int, reg_index_q: int, qc: QuantumCircuit):
        f_qubits = list(qc.qregs[reg_index_q])
        reg = qc.qregs[reg_i_dim]
        for i in reversed(range(len(reg))):
            target = reg[i]
            controls = list(reg[:i]) + f_qubits
            qc.mcx(controls, target)       

    @staticmethod
    def NegativeShift(reg_i_dim: int, reg_index_q: int, qc: QuantumCircuit):
        f_qubits = list(qc.qregs[reg_index_q])
        reg = qc.qregs[reg_i_dim]
        for i in range(len(reg)):
            target = reg[i]
            controls = list(reg[:i]) + f_qubits
            qc.mcx(controls, target)

    @staticmethod
    def CovertIntToBinary(num: int, num_qubit: int) -> list:
        binary = []
        for i in range(num_qubit):
            binary.append((num >> i) & 1)
        return binary

    def FlipCorrespondingQubitForIq(self, reg_index_q: int, num_qubits_for_nq: int, iq: int, qc: QuantumCircuit) -> None:
        qubits_iq = self.CovertIntToBinary(iq, num_qubits_for_nq)
        for idx, bit in enumerate(qubits_iq):
            if bit == 0:
                qc.x(qc.qregs[reg_index_q][idx])

    def Stream(self, qc: QuantumCircuit) -> None:
        lattice_velocity = self.GetLatticeVelocities()

        for iq in range(1, self.nq):
            self.FlipCorrespondingQubitForIq(0, self.num_qubit_f, iq, qc)
            velocity_set = lattice_velocity[:, iq]
            for i_dim in range(0, self.dim):
                if velocity_set[i_dim] == 1:
                    self.PositiveShift(i_dim + 1, 0, qc)
                elif velocity_set[i_dim] == -1:
                    self.NegativeShift(i_dim + 1, 0, qc)
            self.FlipCorrespondingQubitForIq(0, self.num_qubit_f, iq, qc)
            qc.barrier()

    
    @staticmethod
    def MatrixMultiplier(collision_maxtrix: np.ndarray, qc: QuantumCircuit) -> float:
        shape = collision_maxtrix.shape
        if (shape[0] != shape[1]):
            raise ValueError("Collision matrix must be square.")
        num_elements = shape[0]
        num_qubit = int(np.ceil(np.log2(num_elements)))
        num_offset = 2**num_qubit - num_elements;
    
        # SVD
        U, Sigma_tri, Vh = np.linalg.svd(collision_maxtrix)
        max_sigma = np.max(Sigma_tri)
        Sigma = Sigma_tri/ max_sigma

        Vh_ext = np.eye(2**num_qubit, dtype=complex)
        Vh_ext[:shape[0], :shape[1]] = Vh
        U_ext = np.eye(2**num_qubit, dtype=complex)
        U_ext[:shape[1], :shape[0]] = U

        # LCU
        B1 = np.pad(Sigma + 1j * np.sqrt(1 - np.square(Sigma)), (0, num_offset), mode='constant', constant_values=1)
        B2 = np.pad (Sigma - 1j * np.sqrt(1 - np.square(Sigma)), (0, num_offset), mode='constant', constant_values=1)

        q_f = next(reg for reg in qc.qregs if reg.name == 'f')
        q_ancilla = next(reg for reg in qc.qregs if reg.name == 'a')
        coordinate_qubits = []
        for reg_name in ['z', 'y', 'x']:
            reg = next((reg for reg in qc.qregs if reg.name == reg_name), None)
            if reg is not None:
                coordinate_qubits.extend(list(reg))
        q_v = list(q_f) + coordinate_qubits
        
        qc.append(Operator(Vh_ext), q_v[:])

        qc.h(q_ancilla[0])
        for i in range(2**num_qubit):  # Iterate over all basis states
            binary = format(i, f'0{num_qubit}b')
            for j, bit in enumerate(reversed(binary)):
                if bit == '0':
                    qc.x(q_v[j])
            theta1 = np.angle(B1[i])
            qc.x(q_ancilla[0])
            qc.mcp(theta1, q_v[:], q_ancilla[0])
            qc.x(q_ancilla[0])
            theta2 = np.angle(B2[i])
            qc.mcp(theta2, q_v[:], q_ancilla[0])
            for j, bit in enumerate(reversed(binary)):
                if bit == '0':
                    qc.x(q_v[j])
        qc.h(q_ancilla[0])

        qc.append(Operator(U_ext), q_v[:])

        qc.barrier()
        return max_sigma

    
class QcLbmD2Q9 (QcLbm):
    def __init__(self):
        self.dim = 2
        self.nq = 9
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.wi = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        self.num_qubit_f = int(np.ceil(np.log2(self.nq)))
        self.num_max_variables = 2**self.num_qubit_f

    def GetLatticeVelocities(self) -> np.ndarray:
        return np.array([self.cx, self.cy])
    
    @staticmethod
    def GetIqBasedOnDirection(dir: IqDir) -> np.ndarray:
        if dir == IqDir.x_neg:
            return np.array([1, 5, 8])
        elif dir == IqDir.x_pos:
            return np.array([3, 6, 7])
        elif dir == IqDir.y_neg:
            return np.array([4, 7, 8])
        elif dir == IqDir.y_pos:
            return np.array([2, 5, 6])
        else:
            raise ValueError("Invalid direction.")
        
    def ReplacePeriodicVariablesForCollisionMatrix(self, boundary_type: BoundaryType,\
        replacement_matrix: np.ndarray, collision_matrix: np.ndarray):
        if boundary_type == BoundaryType.x_min:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.x_pos)
            ix_req = 0
            ix_anti = self.nx - 1
        elif boundary_type == BoundaryType.x_max:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.x_neg)
            ix_req = self.nx - 1
            ix_anti = 0
        elif boundary_type == BoundaryType.y_min:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.y_pos)
            iy_req = 0
            iy_anti = self.ny - 1
        elif boundary_type == BoundaryType.y_max:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.y_neg)
            iy_req = self.ny - 1
            iy_anti = 0
        else:
            raise ValueError("Invalid boundary type.")
        
        if boundary_type == BoundaryType.y_max or boundary_type == BoundaryType.y_min:
            for ix in range(self.nx):
                node_index_req = iy_req * self.nx * self.num_max_variables + ix * self.num_max_variables
                for index in range(len(index_boundary_req)):
                    node_index_anti = iy_anti * self.nx * self.num_max_variables\
                        + (ix + self.cx[index_boundary_req[index]]) % self.nx * self.num_max_variables
                    for iv in range(self.num_max_variables):
                        collision_matrix[node_index_anti + index_boundary_req[index], node_index_req + iv] =\
                            replacement_matrix[node_index_req + index_boundary_req[index], node_index_req + iv]
                        collision_matrix[node_index_anti + index_boundary_req[index], node_index_anti + iv] = 0

        elif boundary_type == BoundaryType.x_max or boundary_type == BoundaryType.x_min:
            for iy in range(self.ny):
                node_index_req = iy * self.nx * self.num_max_variables + ix_req * self.num_max_variables
                for index in range(len(index_boundary_req)):
                    node_index_anti = ((iy + self.cy[index_boundary_req[index]]) % self.ny) * self.nx * self.num_max_variables\
                        + ix_anti * self.num_max_variables
                    for iv in range(self.num_max_variables):
                        collision_matrix[node_index_anti + index_boundary_req[index], node_index_req + iv] =\
                            replacement_matrix[node_index_req + index_boundary_req[index], node_index_req + iv]
                        collision_matrix[node_index_anti + index_boundary_req[index], node_index_anti + iv] = 0
            
    def SetBounceBackBoundaryForCollisionMatrix(self, boundary_type: BoundaryType, index_u: int, u_0: float,\
        boundary_matrix : np.ndarray, collision_matrix: np.ndarray, initial_field_flattened: np.ndarray):
        if (boundary_matrix.shape != collision_matrix.shape):
            raise ValueError("Boundary matrix must have the same shape as collision matrix.")
        if boundary_type == BoundaryType.y_min:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.y_pos)
            index_boundary_anti = self.GetIqBasedOnDirection(IqDir.y_neg)
            iy_req = 0
        if boundary_type == BoundaryType.y_max:
            index_boundary_req = self.GetIqBasedOnDirection(IqDir.y_neg)
            index_boundary_anti = self.GetIqBasedOnDirection(IqDir.y_pos)
            iy_req = self.ny - 1

        if boundary_type == BoundaryType.y_max or boundary_type == BoundaryType.y_min:
            for ix in range(self.nx):
                node_index_req = iy_req * self.nx * self.num_max_variables + ix * self.num_max_variables
                if np.fabs(u_0) > kEps:
                    initial_field_flattened[node_index_req + index_u] = u_0
                    collision_matrix[node_index_req + index_u, node_index_req + index_u] = 1
                for index in range(len(index_boundary_req)):
                    for iv in range(self.num_max_variables):
                        boundary_matrix[node_index_req + index_boundary_req[index], node_index_req + iv] = \
                            collision_matrix[node_index_req + index_boundary_anti[index], node_index_req + iv]
                    if np.fabs(u_0) > kEps:
                        boundary_matrix[node_index_req  + index_boundary_req[index], node_index_req + index_u] = \
                            -6*self.wi[index_boundary_anti[index]]*self.cx[index_boundary_anti[index]]

    def ComputeCollisionMatrix(self, omega: float) -> np.ndarray:
        self.omega = omega
        collision_matrix = np.zeros((self.num_max_variables*self.nx*self.ny, self.num_max_variables*self.nx*self.ny))
        for j in range(self.ny):
            for i in range(self.nx):
                node_index = j * self.nx * self.num_max_variables + i * self.num_max_variables
                for jq in range(self.nq):
                    for iq in range(self.nq):
                        collision_matrix[node_index + jq, node_index + iq] =\
                            omega * self.wi[jq]* (1 + 3*self.cx[jq]*self.cx[iq] + 3*self.cy[jq]*self.cy[iq])
                        if jq == iq:
                            collision_matrix[node_index + jq, node_index + iq] += (1 - omega)
        return collision_matrix
    
    def AddForceTermToCollisionMatrix(self, index_f: int, force: np.ndarray, collision_matrix: np.ndarray, initial_field_flattened: np.ndarray):
        if force.shape != (2,):
            raise ValueError("Force must be a 2D vector.")
        if (index_f < 0) or (index_f + 1 >= self.num_max_variables):
            raise ValueError("Force index is out of range.")
        for j in range(self.ny):
            for i in range(self.nx):
                node_index = j * self.nx * self.num_max_variables + i * self.num_max_variables
                initial_field_flattened[node_index + index_f] = force[0]
                initial_field_flattened[node_index + index_f + 1] = force[1]
                for iq in range(self.nq):
                    collision_matrix[node_index + iq, node_index  + index_f] = 3 * self.wi[iq] * self.cx[iq]
                    collision_matrix[node_index + iq, node_index  + index_f + 1] = 3 * self.wi[iq] * self.cy[iq]

                collision_matrix[node_index + index_f, node_index + index_f] = 1
                collision_matrix[node_index + index_f + 1, node_index + index_f + 1] = 1


    def FlattenInitialField(self, initial_field: np.ndarray) -> np.ndarray:
        shape = initial_field.shape
        if initial_field.ndim != self.dim + 1:
            raise ValueError(f"Initial field must be {self.dim + 1}D array for {self.dim}D simulation.")
        if shape[-1] > self.num_max_variables:
            raise ValueError(f"Initial field must have at most {self.num_max_variables} variables (2**nq).")
        self.nx, self.ny = shape[:2]
        initial_field_flattened = np.zeros((self.num_max_variables*self.nx*self.ny))
        for iy in range(self.ny):
            for ix in range(self.nx):
                node_index = iy * self.nx * self.num_max_variables + ix * self.num_max_variables
                initial_field_flattened[node_index:node_index + shape[-1]] = initial_field[ix, iy, :]
        return initial_field_flattened