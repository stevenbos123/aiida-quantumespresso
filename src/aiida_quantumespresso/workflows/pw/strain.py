from aiida.engine import WorkChain
from aiida.orm import StructureData, Dict, Str
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.engine import calcfunction

import numpy as np

PwCalculation = CalculationFactory('quantumespresso.pw')
PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
        
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
    elif plane == 'xy+yy':
        strain_matrix[0,1] = strain_value
        strain_matrix[1,1] = 1+ strain_value/3
        
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
        # spec.namespace('strain', help='Inputs for strain application')
        spec.input('strain.structure', valid_type=StructureData)
        spec.input('strain.parameters', valid_type=Dict,
                  help='Dictionary with strain value and plane, e.g., {"value": 0.1, "plane": "xy"}')
            
        spec.expose_inputs(PwBandsWorkChain, namespace='bands')
        spec.output('strained_structure', valid_type=StructureData)
        spec.output('strain_info', valid_type=Dict)
        # spec.output('output_structure', valid_type=StructureData, 
        #            help='Structure after both strain and relaxation')
        spec.expose_outputs(PwBandsWorkChain)
        
        spec.outline(
            cls.validate_inputs,
            cls.apply_strain,
            cls.run_bands,
            cls.get_relaxed_structure
        )
    
    def validate_inputs(self):
        """Validate the input parameters."""
        valid_planes = ['xy', 'yz', 'xz', 'xy+yy']
        params = self.inputs.strain.parameters.get_dict()
        if params['plane'] not in valid_planes:
            self.report(f"Invalid strain plane {params['plane']}")
            return self.exit_codes.ERROR_INVALID_PLANE

    

    def apply_strain(self):
        """Apply the strain using the calcfunction."""
        results = apply_strain(
            structure=self.inputs.strain.structure,
            strain_parameters=self.inputs.strain.parameters
        )
        self.out('strained_structure', results['strained_structure'])
        self.out('strain_info', results['strain_info'])
        self.ctx.strained_structure = results['strained_structure']

    def run_bands(self):
        """Run the PwBandsWorkChain on the strained structure."""
        inputs = self.exposed_inputs(PwBandsWorkChain, namespace='bands')
        inputs['structure'] = self.ctx.strained_structure
        
        running = self.submit(PwBandsWorkChain, **inputs)
        return self.to_context(bands_workchain=running)


    def get_relaxed_structure(self):
        """Extract and output the relaxed structure."""
        # Get the relaxed structure from the bands workchain
        # relaxed_structure = self.ctx.bands_workchain.outputs.output_structure
        # self.out('relaxed_strained_structure', relaxed_structure)
        # self.out('output_structure', relaxed_structure)

        
        # Pass through all other outputs from bands calculation
        self.out_many(
            self.exposed_outputs(
                self.ctx.bands_workchain, 
                PwBandsWorkChain
            )
        )