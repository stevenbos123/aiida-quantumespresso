from aiida.engine import WorkChain
from aiida.orm import StructureData, Dict, Str
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.engine import calcfunction

import numpy as np

PwCalculation = CalculationFactory('quantumespresso.pw')

        
@calcfunction
def apply_strain(structure: StructureData, strain_parameters: Dict) -> dict:
    """Apply strain to the structure."""
    ase_structure = structure.get_ase()
    params = strain_parameters.get_dict()
    strain_value = params['value']
    plane = params['plane']
    
    # Create strain matrix based on plane
    strain_matrix = np.eye(3)
    if plane == 'xy':
        strain_matrix[0,1] = strain_value
    elif plane == 'yz':
        strain_matrix[1,2] = strain_value
    elif plane == 'xz':
        strain_matrix[0,2] = strain_value
        
    cell = ase_structure.get_cell()
    new_cell = np.dot(cell, strain_matrix)
    ase_structure.set_cell(new_cell, scale_atoms=True)
    
    return {
        'strained_structure': StructureData(ase=ase_structure),
        'strain_info': Dict(dict={
            'plane': plane,
            'value': strain_value,
            'matrix': strain_matrix.tolist()
        })
    }

class ShearStrainWorkChain(WorkChain):
    _process_class = PwCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # spec.expose_inputs(PwCalculation, namespace='pw')
        spec.input('structure', valid_type=StructureData)
        spec.input('strain_parameters', valid_type=Dict, 
                  help='Dictionary with strain value and plane, e.g., {"value": 0.1, "plane": "xy"}')        
        spec.output('strained_structure', valid_type=StructureData)
        spec.output('strain_info', valid_type=Dict)
        
        spec.outline(
            cls.validate_inputs,
            cls.apply_strain,
        )
    
    def validate_inputs(self):
        """Validate the input parameters."""
        valid_planes = ['xy', 'yz', 'xz']
        params = self.inputs.strain_parameters.get_dict()
        if params['plane'] not in valid_planes:
            self.report(f"Invalid strain plane {params['plane']}")
            return self.exit_codes.ERROR_INVALID_PLANE

    

    def apply_strain(self):
        """Apply the strain using the calcfunction."""
        results = apply_strain(
            structure=self.inputs.structure,
            strain_parameters=self.inputs.strain_parameters
        )
        self.out('strained_structure', results['strained_structure'])
        self.out('strain_info', results['strain_info'])
