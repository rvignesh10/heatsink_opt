"""
Pybind11 wrapper for HeatSinkSolver with Dictionary Initialization
"""
from __future__ import annotations
import numpy
import numpy.typing

__all__: list[str] = ['HeatSinkSolver', 'finalize_mpi', 'init_mpi', 'getCharProps']

def getCharProps(ref: str, T_in: float, p_out: float, L_c: float) -> dict:
    """takes in refrigerant name, inlet temp, outlet pressure and characteristic length as inputs

    Returns:
        dict: char_props that contains all characteristic properties
    """
    ...

class HeatSinkSolver:
    def __init__(self, config: dict) -> None:
        """
        Initialize HeatSinkSolver using a nested Python dictionary
        """
        ...

    #########################################################################################################################
    
    def calcAggregateOutletVaporQuality(self, state: numpy.typing.NDArray[numpy.float64]) -> float:
        """
        returns aggregated vapor quality at the outlet of the heat sink
        """
        ...

    def calcAggregateOutletVaporQualityJacobian(self, state: numpy.typing.NDArray[numpy.float64], dxagg_dstate: numpy.typing.NDArray[numpy.float64]) -> None:
        """
        calculates the jacobian of xagg w.r.t heat sink states
        """
        ...
    
    def diffAggregateOutletVaporQuality(self) -> float:
        """
        returns 0.0 as aggregated quality is not a function of design variables
        """
        ...
    
    #########################################################################################################################
    
    def calcNDPumpingPower(self, state: numpy.typing.NDArray[numpy.float64]) -> float:
        """
        returns the non-dimensional pumping power for a specified pressure-drop and vol-flow rate through the heat sink
        """
        ...
    
    def calcNDPumpingPowerJacobian(self, state: numpy.typing.NDArray[numpy.float64], dpower_dstate: numpy.typing.NDArray[numpy.float64]) -> None:
        """
        calculates the jacobian of non-dimensional pump power w.r.t heat sink states
        """
        ...
    
    def diffNDPumpingPower(self, state: numpy.typing.NDArray[numpy.float64], flow_bcs_star_bar: float) -> float:
        """
        returns partial gradient of non-dimensional pumping power w.r.t des variables
        """
        ...
    
    #########################################################################################################################
        
    def calcAggregatedSourceTemperature(self, state: numpy.typing.NDArray[numpy.float64], boundary_att_id: int=1) -> float:
        """
        returns aggregated temperature at a given boundary attribute
        """
        ...
    
    def calcAggregateSourceTemperatureJacobian(self, state: numpy.typing.NDArray[numpy.float64], dTagg_star_dstate: numpy.typing.NDArray[numpy.float64], boundary_att_id: int=1) -> None:
        """
        calculates the jacobian of Tagg w.r.t state variables
        """
        ...

    def diffAggregateSourceTemperature(self) -> float:
        """
        returns 0.0 as Tagg is not dependent on design variables
        """
        ...

    #########################################################################################################################
    
    def calcModifiedThermalResistance(self, state: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, alpha: float=1.0, beta: float=1.0e+03, boundary_att_id: int=1) -> float:
        """
        returns (Tagg - Tin) * ( Qdot * delta_p )^gamma : as the modified thermal resistance
        """
        ...
    
    def calcModifiedThermalResistanceJacobian(self, state: numpy.typing.NDArray[numpy.float64], dmtr_star_dstate: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, alpha: float=1.0, beta: float=1.0e+03, boundary_att_id: int=1) -> None:
        """
        calculates the jacobian of the modified thermal resistance w.r.t state variables
        """
        ...
    
    def diffModifiedThermalResistance(self, state: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, flow_bcs_star_bar: float, alpha: float=1.0, beta: float=1.0e+03, boundary_att_id: int=1) -> float:
        """
        calculates the total derivative of mod_Rth w.r.t alpha_p
        """
        ...
    
    #########################################################################################################################
    
    def calcHeatSinkOutputs(self, state: numpy.typing.NDArray[numpy.float64]) -> tuple[float, float]:
        """
        returns non-dimensional thermal-flux and normalized thermal-flux as a tuple
        """
        ...

    def calcHeatSinkOutputsJacobian(self, state: numpy.typing.NDArray[numpy.float64], dflux_star_dstate: numpy.typing.NDArray[numpy.float64], dnormFlux_star_dstate: numpy.typing.NDArray[numpy.float64]) -> None:
        """
        sets the jacobian of the outputs w.r.t heat sink state variables
        """
        ...

    def diffHeatSinkOutputs(self, state: numpy.typing.NDArray[numpy.float64], flow_bcs_star_bar: float) -> tuple[float, float]:
        """
        returns the partials of outputs w.r.t design variables as a tuple
        """
        ...
    
    #########################################################################################################################
    
    def calcNDPumpSpeed(self, state: numpy.typing.NDArray[numpy.float64]) -> float:
        """
        returns the non-dimensional rpm of the pump for a specified pressure-drop and vol-flow rate through the heat sink
        """
        ...

    def calcNDPumpSpeedJacobian(self, state: numpy.typing.NDArray[numpy.float64], dNstar_dstate: numpy.typing.NDArray[numpy.float64]) -> None:
        """
        calculates the jacobian of non-dimensional pump speed w.r.t heat sink states
        """
        ...

    def diffNDPumpSpeed(self, state: numpy.typing.NDArray[numpy.float64], flow_bcs_star_bar: float) -> float:
        """
        returns the partial gradient of non-dimensional pump speed w.r.t des variables
        """
        ...
    
    #########################################################################################################################
    
    def calcPressureDropConstraint(self, state: numpy.typing.NDArray[numpy.float64]) -> float:
        """
        return the pressure-drop-heatsink - pressure-drop-available
        """
        ...
    
    def calcPressureDropConstraintJacobian(self, state: numpy.typing.NDArray[numpy.float64], dpdcons_dstate: numpy.typing.NDArray[numpy.float64]) -> None:
        """
        calculates the jacobian of non-dimensional pressure-drop-constraint w.r.t heat sink states
        """
        ...    

    def diffPressureDropConstraint(self, state: numpy.typing.NDArray[numpy.float64], flow_bcs_star_bar: float) -> float:
        """
        returns the gradient of pressure-drop-constraint w.r.t des variables
        """
        ...
    
    #########################################################################################################################
    
    def calcSafeSourceTemperature(self, state: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, alpha: float, boundary_att_id: int=1) -> float:
        """
        calculates a combination of temp uniformity and temp target
        """
        ...
    
    def calcSafeSourceTemperatureJacobian(self, state: numpy.typing.NDArray[numpy.float64], dJ_dstate: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, alpha: float, boundary_att_id: int=1) -> None:
        """
        calculates the jacobian of safe source temperature obj w.r.t state variables
        """
        ...
    
    def diffSafeSourceTemperature(self, state: numpy.typing.NDArray[numpy.float64], Tsrc_star: float, flow_bcs_star_bar: float, alpha: float, boundary_att_id: int=1) -> float:
        """
        returns 0.0 as its not a function of design variables at all
        """
        ...
    
    #########################################################################################################################
    
    def getCharProps(self) -> dict:
        """
        Returns the characteristic properties as a dictionary
        """
        ...

    def getTotalNumStates(self) -> int:
        """
        returns the total number of state variables
        """
        ...

    def SaveToParaview(self, state: numpy.typing.NDArray[numpy.float64], tcollection: str, fcollection: str) -> None:
        """
        saves the solution state as a .pvd file that can be viewed through ParaView
        """
    
    def setInputs(self, flow_loads: list[dict], thermal_load: dict) -> None:
        """
        Set inputs using a list of flow dictionaries and one thermal dictionary
        """
        ...

    def set_thermal_bcs(self, source_bcs_star: float) -> None:
        """
        sets the source dirichlet temperature or heat flux boundary condition for the thermal solver
        """
        ...
    
    def set_flow_bcs(self, flow_bcs_star: float) -> None:
        """
        sets the flow solver boundary condition - either inlet pressure des var or mass-flow/channel des var
        """
        ...

    def solveForAdjoint(self, state: numpy.typing.NDArray[numpy.float64], dJdstate: numpy.typing.NDArray[numpy.float64], adjoint: numpy.typing.NDArray[numpy.float64], use_eigen_solver: bool = False) -> None:
        """
        Solves the adjoint equation (dRdu)^T * lambda = -dJdu
        """
        ...

    def solveForState(self, state: numpy.typing.NDArray[numpy.float64], use_eigen: bool=False, prec_solve: bool=True) -> None:
        """
        solves the forward problem R(u)=0
        """
        ...

    def calcHeatSinkTotalGradient(self, state: numpy.typing.NDArray[numpy.float64], adjoint: numpy.typing.NDArray[numpy.float64], flow_bcs_star_bar: float, vxyz_star_bar: numpy.typing.NDArray[numpy.float64], xi_star_bar: numpy.typing.NDArray[numpy.float64], g_star_bar: numpy.typing.NDArray[numpy.float64], J_bar: float) -> float:
        """
        returns the total gradient of the Lagrangian L(u, lambda, alpha), dL_dalpha = J_bar + lambda^T * R_bar
        """
        ...
    
def finalize_mpi() -> None:
    """
    Finalize the MPI environment
    """
    ...

def init_mpi() -> None:
    """
    Initialize the MPI environment (required for MFEM)
    """
    ...