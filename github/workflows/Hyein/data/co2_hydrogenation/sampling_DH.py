
from config_STEAM import PARAMS, get_REACT
from optimize import run_optimization
from utils import kill_processes


VARS = {# (var, type, lb, ub)
    'DH': [
        ('C103.discharge_pressure', float, 50.0, 100.0),
        ('C301.discharge_pressure', float, 50.0, 100.0),
        ('HX100.value', float, None, None),
        ('E110.temperature', float, 200, 250),
        ('R100.tube_length', float, 3, 10),
        ('R100.tube_diameter', float, 0.5, 5),
        ('S100.split_fractions.115', float, 0.1, 0.9),  ###7
        ('E120.temperature', float, 25, 100),
        ('VLV140.outlet_pressure', float, 1.0, 3.0),  ###9
        ('P150.discharge_pressure', float, 1.0, 10.0),
        ('HX150.value', float, 50.0, 150.0),
        ('HX220.value', float, 150.0, 250.0),
        ('E220.temperature', float, 200.0, 350.0),
        ('D200.reflux_ratio', float, 0.1, 3.0),
        ('D200.condenser_pressure', float, 0.1, 1.0),
        ('S220.split_fractions.H2O-4', float, 0.1, 0.9), ###16
    ]
}


OBJECTIVES = {
    'minimum_selling_price': {
        'alias': 'MSP',
        'unit': '[$/kg]',
        'objective': 'minimization',
        'scaling': 1,
        },
    'CO2_emission': {
        'alias': 'GWP',
        'unit': '[kg $CO_{2}e$/kg]',
        'objective': 'minimization',
        'scaling': 2,
        },
}


CASE = {
    'DH': ('capture_mea_lower_duty_v1', 'direct_hydrogenation_v8_heat_integration_corrected'),
}    


if __name__ == '__main__':
    project = 'STEAM'
    kill_processes('AspenPlus.exe')
    
    for case, filenames in CASE.items():
        variables = VARS[case]
        
        PARAMS['REACT'] = get_REACT(filenames[1])
        
        record = run_optimization(project, filenames, variables, target_product='MEOH', 
                                  OBJECTIVES=OBJECTIVES, PARAMS=PARAMS, method='NSGA2', 
                                  n_iter=300, n_line_search=10, monitor=False, resume=True, parallel=True)