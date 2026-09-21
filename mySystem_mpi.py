import argparse, psutil, cProfile, mmap, math, os, tracemalloc, time, re,sys
import numpy as np
import MDAnalysis as mda
from collections import Counter
from mpi4py import MPI
from operator import itemgetter

class system():
    
    def __init__(self,config:"configuration"):

        self.CONFIG = config

        tracemalloc.start()

        self.COMM = MPI.COMM_WORLD
        self.NP = self.COMM.Get_size()
        self.RANK = self.COMM.Get_rank()
        self.PROCESS = psutil.Process(os.getpid())
        self.CURRENT_LOCAL_MEMORY_USE = 0 #Current process-specific mem use in Bytes
        self.PEAK_MEMORY_USE = None #Peak memory use (of all processes) in GB (only updated on root)

        if self.RANK == 0:
            self.CONFIG.STDOUT.write('Creating system...\n')

        self.DATA_FILE = None
        self.TRJ_FILE = None
        self.TOPO_FILE = None

        #Timing and processing rate variables
        self.START_TIME = None #Run start time
        self.PROCESSING_RATE = None #Current processing rate in ns/day
        if self.RANK == 0:
            self.PEAK_MEMORY_USE = 0
            self.START_TIME = time.time()

        self.ELEMENTS = {
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
            "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
            "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        }

        self.PASS = 1 #Universal proceed flag

        self.line = '' #Current file line

        #system cell box lengths
        self.boxLengthX = 0
        self.boxLengthY = 0
        self.boxLengthZ = 0 

        #data file molecule parameters
        self.NUM_ATOMS = 0
        self.NUM_ATOM_TYPES = 0
        self.NUM_BONDS = 0
        self.NUM_BOND_TYPES = 0
        self.NUM_ANGLES = 0
        self.NUM_ANGLE_TYPES = 0
        self.NUM_DIHEDRALS = 0
        self.NUM_DIHEDRAL_TYPES = 0
        self.NUM_IMPROPERS = 0
        self.NUM_IMPROPER_TYPES = 0

        #Atom-based data structures
        self.ATOMS = None
        self.ATOMTYPES = None
        self.ELEMENTSYMBOLS = None
        self.MOLTYPES = None
        self.CHARGES = None
        self.X = None
        self.Y = None
        self.Z = None
        self.boxX = []
        self.boxY = []
        self.boxZ = []
        self.ATOM_DIELECTRICS = None

        #Bonding info
        self.BONDTYPES = []

        #Molecule-based data structures
        self.CONNECTIVITY = None #Array that stores (ordered or unordered) molecular connectivity via molecule indexes stored under atomic indexes (ex. [33,33,33,4,4,16,16,16,...])
        self.MOLECULES = None #Molecule object array
        self.NUM_MOLECULES = 0 #Molecule number

        self.POINTPARTICLESX = None
        self.POINTPARTICLESY = None
        self.POINTPARTICLESZ = None
        self.ALL_POINTPARTICLESXYZ = None

        #Reactant molecules of interest for analysis
        self.REACTANTS = None
        self.NUM_REACTANTS = 0
        self.CUTOFF_TARGETS = None

        #Dynamic dictionary containing the count of each conserved residue within the current reactant cluster
        self.CURRENT_CONSERVED_RESIDUES_TARGETS = {}
        #Dynamic dictionary containing the count overage of each conserved residue within the current reactant cluster
        self.CURRENT_CONSERVED_RESIDUES_COUNTS = {}
        
        #Output variables
        self.graphOut = None
        self.debugOutFiles = None

        #Trajectory generation and indexing
        self.FRAME_GENERATOR = None
        self.TIMESTEP = 0
        self.START_STEP = 0
        self.STEP = None
        self.START_FRAME = 0
        self.STOP_FRAME = None
        self.FRAME_STRIDE = 1

        #Reaction graph object (see rxnGraph class)
        self.GRAPH = rxnGraph(self.CONFIG)
        self.GRAPH2 = None #For secondary graphs when reading user reaction graph files
        self.ALL_RANKS_GRAPH = rxnGraph(self.CONFIG)
        self.REACTANT_TO_PRINT = None

        #Extraction statistical variables
        self.RANK_CLUSTERS_TO_EXTRACT = self.CONFIG.config['CLUSTERS_TO_EXTRACT']
        self.TOTAL_CLUSTERS_TO_EXTRACT = self.CONFIG.config['CLUSTERS_TO_EXTRACT']
        self.RANK_CLUSTERS_WRITTEN = 0
        self.TOTAL_CLUSTERS_WRITTEN = 0
        self.REACTANTS_TO_WRITE_FOUND = 0
        self.TOTAL_REACTANTS_TO_WRITE_FOUND = 0

        #All atom gromacs indexes set for making .ndx files
        self.GMX_SYSTEM_INDEXES = None
        
        #Extraction failure statistics dict:
        self.EXTRACT_FAIL_STATS = {
            #Not enough time has elapsed since last extraction:
            'WRITE_FAILED_OVER_EXTRACT_FREQ':0,
            'TOTAL_WRITE_FAILED_OVER_EXTRACT_FREQ':0,
            #The inner solvation shells contain an excluded residue:
            'WRITE_FAILED_INNER_EXCLUDE_RES':0,
            'TOTAL_WRITE_FAILED_INNER_EXCLUDE_RES':0,
            ##The inner solvation shells contain a non-requested residue:
            'WRITE_FAILED_INNER_NONCONSERVE':0,
            'TOTAL_WRITE_FAILED_INNER_NONCONSERVE':0,
            #The inner solvation shells + ligand conserved 1st shell contain too many of a conserved residue:
            'WRITE_FAILED_FIRSTOUTER_NONCONSERVE':0,
            'TOTAL_WRITE_FAILED_FIRSTOUTER_NONCONSERVE':0,
            #The first solvation shell of a conserved ligand contains more than conserved solvent residues (such as a spectator ion):
            'WRITE_FAILED_FIRSTOUTER_NONSOLVENT':0,
            'TOTAL_WRITE_FAILED_FIRSTOUTER_NONSOLVENT':0,
            #Upon inclusion of the outer solvation shells, the cluster is contains too few conserved residues:
            'WRITE_FAILED_SECONDOUTER_UNDERCOORD':0,
            'TOTAL_WRITE_FAILED_SECONDOUTER_UNDERCOORD':0,
            #Cluster hydrogen bonding is outside requested range: 
            'WRITE_FAILED_OVER_HBOND_COUNT':0,
            'TOTAL_WRITE_FAILED_OVER_HBOND_COUNT':0,
            'WRITE_FAILED_UNDER_HBOND_COUNT':0,
            'TOTAL_WRITE_FAILED_UNDER_HBOND_COUNT':0}

        # self.extractCoordOut = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/rank_{self.RANK}_coord.txt",'w')
        # self.failedExtractCoordOut = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/rank_{self.RANK}_failed_coord.txt",'w')

        #Writing timestep information fix
        self.EXTRACTION_TIMESTEP = 20000 #timestep for writing clusters in femtoseconds
        self.GRAPH_WRITE_TIMESTEP = 5000 #timestep for writing reaction graph in femtoseconds
        self.CLUSTER_WRITE_OFFSET = 0
        if (self.TOTAL_CLUSTERS_TO_EXTRACT > 0) and (self.TOTAL_CLUSTERS_TO_EXTRACT != -1):
            for rank in range(self.RANK):
                self.CLUSTER_WRITE_OFFSET += ((self.TOTAL_CLUSTERS_TO_EXTRACT // self.NP) + int((self.TOTAL_CLUSTERS_TO_EXTRACT % self.NP) >= rank+1))
            self.RANK_CLUSTERS_TO_EXTRACT = (self.TOTAL_CLUSTERS_TO_EXTRACT // self.NP) + int((self.TOTAL_CLUSTERS_TO_EXTRACT % self.NP) >= self.RANK+1)

    #System inner class- molecule

    class molecule():
        def __init__(self, configuration:"configuration", atom_indexes=None, index:int=-1):
            
            self.CONFIG = configuration
            self.index = index
            self.atoms = [] #indexes of atoms in molecule
            self.residue = None #Molecule type

            self.cluster = None #Molecular cluster formed around molecule
            self.coordination = [{} for shell in range(self.CONFIG.config['REACTION_SHELLS'])]
            self.solvation_shells = [] #Data structure describing solvation: [[firstshell],[secondshell],...]
            self.total_shells = None
            self.extract_shells = None #Subset of solvation_shells, the portion of the solvation shells which may be extracted
            self.extract_shells_hbonds = 0
            self.atom_coordinations = {} #Dictionary storing molecular coordination for each atom index, {atom_index:set(molecule_index1,molecule_index2,...)} 
            self.atom_hbond_donation_atoms = {} #Molecule indexes donated hydrogen bonds from this molecule
            self.outer_shell = 0

            self.current = None #Current coordination state ref. to rxnGraph node, allows for reaction edges to be drawn simetaneously for all species traversing graph
            self.extract_wait_steps = 0 #Extraction timer, tracks the number of steps until extraction may occur again
            
            self.cluster_atoms = None
            self.cluster_mols = None
            self.atom_type_masks = None
            self.remaining_molecule_mask = None

            if isinstance(atom_indexes,int):
                self.atoms.append(atom_indexes)
                self.atom_coordinations[atom_indexes] = set()
                self.atom_hbond_donation_atoms[atom_indexes] = set()
            elif isinstance(atom_indexes,list):
                atom_indexes_set = set(atom_indexes)
                if len(atom_indexes) != len(atom_indexes_set):
                    raise mySystemError("Redundant atom list provided to molecule()")
                else:
                    for atom_index in atom_indexes:
                        self.atom_coordinations[atom_index] = set()
                        self.atom_hbond_donation_atoms[atom_index] = set()

                    self.atoms.extend(atom_indexes)
            else:
                raise mySystemError("Invalid atom parameters provided for molecule()")
            
            self.atoms = np.array(self.atoms)

        def addAtom(self, atom_index:int):
            if not isinstance(atom_index,int):
                raise mySystemError("Invalid parameters provided for molecule.addAtom")
            
            self.atoms = np.append(self.atoms,atom_index)
            self.atom_coordinations[atom_index] = set()
            self.atom_hbond_donation_atoms[atom_index] = set()

        def updateSolvationShells(self,shell_index,new_contents):
            self.solvation_shells.extend([set() for _ in range((shell_index+1)-len(self.solvation_shells))])
            if isinstance(new_contents,set):
                self.solvation_shells[shell_index].update(new_contents)
                return
            elif isinstance(new_contents,np.int32):
                self.solvation_shells[shell_index].add(new_contents)
                return
            raise mySystemError('molecule.updateSolvationShells cannot take non-set or integer argument')

        def resetSolvationShells(self):
            for shell in self.solvation_shells:
                shell.clear()

            for atom_key in self.atom_coordinations:
                self.atom_coordinations[atom_key].clear()

            self.total_shells = None

            self.extract_shells = None

            for atom_key in self.atom_hbond_donation_atoms:
                self.atom_hbond_donation_atoms[atom_key].clear()

            self.extract_shells_hbonds = 0

        def buildTotalShells(self):
            self.total_shells = {self.index} | {mol for shell in self.solvation_shells for mol in shell} 

        def enumerateCluster(self,MOLECULES):

            self.cluster_mols = []
            self.cluster_atoms = []

            for molecule in self.cluster:
                for atom in MOLECULES[molecule].atoms:
                    self.cluster_mols.append(molecule)
                    self.cluster_atoms.append(atom)

            self.cluster_mols = np.array(self.cluster_mols, dtype=np.int32)
            self.cluster_atoms = np.array(self.cluster_atoms, dtype=np.int32)

        def createTypeMasks(self,ATOM_TYPES):
        
            self.atom_type_masks = {}
            cluster_atom_types = ATOM_TYPES[self.cluster_atoms]
            for a_type in self.CONFIG.config['ATOM_TYPE_LIST']:
                self.atom_type_masks[a_type] = (cluster_atom_types == a_type)

        def createRemainingMolMask(self):
            self.remaining_molecule_mask = np.ones(len(self.cluster_mols), dtype=bool)

        def __str__(self):
            return ('[' + str(self.index) + ']: (' + str(self.residue) + ') atoms- ' + str(self.atoms) + ' cluster molecules- ' + str(self.cluster))

    #System class methods:

    def checkConfiguration(self,parameter_name,optional_temp_param=None):
        PARAMETER = self.CONFIG.config[parameter_name]
        SYNTAX = self.CONFIG.syntax[parameter_name]

        match parameter_name:
            case 'SYSTEM_TYPE':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER != 'gromacs' and PARAMETER != 'lammps':
                    raise configurationError(f"Invalid input for '{parameter_name}': {PARAMETER}")
            case 'DATA_FILE_PATH':
                try:
                    self.DATA_FILE = open(PARAMETER, 'r')
                except FileNotFoundError:
                    raise configurationError(f"Error: File not found: {PARAMETER}")
            case 'TRJ_FILE_PATH':
                try:
                    self.TRJ_FILE = open(PARAMETER, 'r')
                except FileNotFoundError:
                    raise configurationError(f"Error: File not found: {PARAMETER}")
            case 'TOPOLOGY_FILE_PATH':
                if PARAMETER is None:
                    return
                try:
                    self.TOPO_FILE = open(PARAMETER, 'r')
                except FileNotFoundError:
                    return #Topology file is optional, return False
            case 'DUMP_FREQ':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER <= 0:
                    raise configurationError(f"'{parameter_name}' outside valid range, should be > 0")
            case 'FRAMES_TO_PROCESS':
                if PARAMETER is None:
                    raise mySystemError(f"'{parameter_name}' missing")
                if isinstance(PARAMETER,int):
                    self.CONFIG.config[parameter_name] = (PARAMETER,)
                    PARAMETER = self.CONFIG.config[parameter_name]
                if isinstance(PARAMETER,list):
                    self.CONFIG.config[parameter_name] = tuple(PARAMETER)
                    PARAMETER = self.CONFIG.config[parameter_name]
                for slice_arg in PARAMETER:
                    if not isinstance(slice_arg,int):
                        raise mySystemError(f"Non-integer slicing argument provided for '{parameter_name}'")
                if len(PARAMETER) == 1:
                    if PARAMETER[0] < 0 and PARAMETER[0] != -1:
                        raise configurationError(f"Invalid '{parameter_name}', requested frame processing limit must be integer > 0 or -1 for all frames")
                    self.STOP_FRAME = PARAMETER[0]
                    return
                if len(PARAMETER) > 3:
                    raise mySystemError(f"Too many arguments for '{parameter_name}' slice. Syntax: {SYNTAX}")
                if PARAMETER[0] > PARAMETER[1]:
                    raise mySystemError(f"Invalid slice for '{parameter_name}': start index exceeds stop index. Syntax: {SYNTAX}")
                
                self.START_FRAME = PARAMETER[0]
                self.STOP_FRAME = PARAMETER[1]
                if len(PARAMETER) > 2:
                    if PARAMETER[2] < 1:
                        raise mySystemError(f"Invalid slice for '{parameter_name}': invalid stride. Syntax: {SYNTAX}")
                    self.FRAME_STRIDE = PARAMETER[2]
            case 'ATOM_TYPE_LIST':
                if self.NUM_ATOM_TYPES is None:
                    raise configurationError('Number of atom types is ambiguous')
                if PARAMETER is None and optional_temp_param is None:
                    raise configurationError(f"'Atom types could not be parsed from the file and no {parameter_name} was provided")
                if PARAMETER is not None:
                    if len(PARAMETER) != self.NUM_ATOM_TYPES:
                        raise configurationError(f"'{parameter_name}' ({PARAMETER}) length does not match atom type number parsed from file")
                    if self.RANK == 0:
                        self.CONFIG.STDOUT.write(f"'{parameter_name}' accepted\n")
                if optional_temp_param is not None:
                    if PARAMETER is not None:
                        if optional_temp_param == PARAMETER:
                            if self.RANK == 0:
                                self.CONFIG.STDOUT.write(f"WARNING: redundant '{parameter_name}' provided, this parameter is option for this run style.\n")
                        else:
                            raise configurationError(f"Optional '{parameter_name}' provided for GROMACS type system does not match that parsed from data file")
                    else:
                        self.CONFIG.config[parameter_name] = optional_temp_param

                if all([isinstance(self.ATOMTYPES[0],int) for _ in self.ATOMTYPES]): #Check that an all strings atom type list exists
                    for i in range(len(self.ATOMTYPES)):
                        for j in range(len(PARAMETER)):
                            type = PARAMETER[j]
                            if self.ATOMTYPES[i] == j+1:
                                self.ATOMTYPES[i] = type
                elif not all([isinstance(self.ATOMTYPES[0],str) for _ in self.ATOMTYPES]):
                    raise configurationError('Could not parse atom types from data file')

                if not isinstance(self.ATOMTYPES, np.ndarray):
                    self.ATOMTYPES = np.array(self.ATOMTYPES)
            case 'RESIDUE_LIST':
                if PARAMETER is None and optional_temp_param is None:
                    raise configurationError(f"'Residue types could not be parsed from data file and no {parameter_name} was provided")
                if PARAMETER is not None:
                    if optional_temp_param is not None and optional_temp_param == PARAMETER:
                        if self.RANK == 0:
                            self.CONFIG.STDOUT.write(f"WARNING: redundant '{parameter_name}' provided, this parameter is option for this run style.\n")
                    else:
                        residue_atomtype_lists_sorted = []
                        for residue,atom_type_list in PARAMETER.items():
                            #Check for list validity
                            if not isinstance(atom_type_list,list):
                                raise configurationError(f"Invalid atom list given in '{parameter_name}' for residue {residue}. Syntax is {SYNTAX}")
                            sorted_entry = sorted(atom_type_list)
                            #Check for redundant entries
                            if sorted_entry in residue_atomtype_lists_sorted:
                                raise configurationError(f"Duplicate residue definition encounterd, '{residue}' : '{atom_type_list}'")
                            residue_atomtype_lists_sorted.append(sorted_entry)
                            #Check for atom_type validity
                            # for atom_type in atom_type_list:
                            #     if atom_type not in self.CONFIG.config['ATOM_TYPE_LIST']: #Check if all entries and atom types within entries are good
                            #         raise configurationError(f"Invalid atom type '{atom_type}'")
                            #Check that for each residue definiton there is >=1 matching molecule
                            if self.MOLECULES is None:
                                raise configurationError(f"No molecules present at time of {parameter_name} validation")
                            mol_present = False
                            for molecule in self.MOLECULES:
                                if sorted(self.ATOMTYPES[molecule.atoms]) == sorted_entry:
                                    mol_present = True
                            if not mol_present:
                                if self.RANK == 0:
                                    self.CONFIG.STDOUT.write(f"WARNING: '{parameter_name}' - '{residue}' is not used.\n")
                        #Check that for each molecule there is 1 residue type to describe its atoms
                        for molecule in self.MOLECULES:
                            if sorted(self.ATOMTYPES[molecule.atoms]) not in residue_atomtype_lists_sorted:
                                raise configurationError(f"'{parameter_name}' is incomplete as provided, missing definition for {self.ATOMTYPES[molecule.atoms]}")
                    if self.RANK == 0:
                        self.CONFIG.STDOUT.write(f"'{parameter_name}' accepted\n")
                if optional_temp_param is not None and PARAMETER is None:
                    self.CONFIG.config[parameter_name] = optional_temp_param
            case 'REACTANT':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER not in self.CONFIG.config['RESIDUE_LIST']:
                    raise configurationError('Invalid reactant residue type provided')
            case 'REACTION_SHELLS':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER < 1:
                    raise configurationError(f"'{parameter_name}' outside valid range, should be > 0")
            case 'CLUSTER_MOLECULES':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER < 1:
                    raise configurationError(f"'{parameter_name}' outside valid range, should be > 0")
            case 'PAIR_CUTOFFS':
                def strInAtomTypeList(string):
                    if string not in self.CONFIG.config['ATOM_TYPE_LIST']:
                        raise configurationError(f"Encountered '{parameter_name}' atom name,{string}, not in 'ATOM_TYPE_LIST'")
                if not PARAMETER:
                    raise configurationError(f"Empty {parameter_name} provided")
                for key,cutoff in PARAMETER.items():
                    #Check if key is a atom pair tuple
                    if not isinstance(key,tuple):
                        raise configurationError(f"Invalid {parameter_name} pair given: {key}, syntax is {SYNTAX}")
                    #Check that tuple is of length 2, specifying cutoff between two atom types or atom_type groups
                    if len(key) > 2:
                        raise configurationError(f"'{parameter_name}' between greater than two atoms found (e.g. (A,B,C):cutoff), use tuples to signify cutoffs between more than two atom types or atom type groups, e.g ((A,B),(C,D)):X.XX or ((A,B),C):X.XX")
                    if not isinstance(cutoff,float) and not isinstance(cutoff,int):
                        raise configurationError(f"Non-numerical pair cutoff encounterd in '{parameter_name}': {cutoff}, syntax is {SYNTAX}")
                    for pair in key:
                        if isinstance(pair,str):
                            pass
                            #strInAtomTypeList(pair)
                        elif isinstance(pair,tuple):
                            for nested_pair in pair:
                                if not isinstance(nested_pair,str):
                                    raise configurationError(f"Non-string atom name encounterd in nested '{parameter_name}' tuple: {nested_pair}, syntax is {SYNTAX}")
                                #strInAtomTypeList(nested_pair)
                        else:
                            raise configurationError(f"Non-string/tuple atom name(s) encounterd in '{parameter_name}' tuple: {pair}, syntax is {SYNTAX}")
                
                #Create atom-type specific list of relevent target atom types from self.CONFIG.config['PAIR_CUTOFFS']
                self.completeCutOffList()
                PARAMETER = self.CONFIG.config[parameter_name] #Reassign the PARAMETER variable, as completeCutOffList() alters self.CONFIG.config['PAIR_CUTOFFS']

                #Check if there is at least 1 atom type cutoff between each residue pair
                residue_items = list(self.CONFIG.config['RESIDUE_LIST'].items())
                res_num = len(self.CONFIG.config['RESIDUE_LIST'].items())
                for residue_index1 in range(res_num):
                    residue1,atom_type_list1 = residue_items[residue_index1]
                    for residue_index2 in range(residue_index1+1,res_num):
                        residue2,atom_type_list2 = residue_items[residue_index2]
                        pair_found = False
                        for atom_type1 in atom_type_list1:
                            for atom_type2 in atom_type_list2:
                                if (atom_type1,atom_type2) in PARAMETER:
                                    pair_found = True
                                    break
                            if pair_found:
                                break
                        if not pair_found:
                            if self.RANK == 0:
                                self.CONFIG.STDOUT.write(f"WARNING: atom pair interaction cutoff not found between residues '{residue1}' and '{residue2}'\n")
            case 'CREATE_RXN_GRAPH':
                if PARAMETER is None:
                    self.CONFIG.config[parameter_name] = False
            case 'WRITE_DIRECTORY':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER == '':
                    raise mySystemError(f"Empty {parameter_name} given")
            case 'RUN_NAME':
                if PARAMETER is None:
                    raise mySystemError(f"{parameter_name} is None")
                if PARAMETER == '':
                    raise mySystemError(f"Empty {parameter_name} given")
            case 'OUTPUT_TYPE':
                if PARAMETER is None:
                    if self.CONFIG.config['CLUSTERS_TO_EXTRACT'] == -1 or self.CONFIG.config['CLUSTERS_TO_EXTRACT'] > 0:
                        raise configurationError(f"No '{parameter_name}' style provided to extract clusters to")
                    return
                check_param = PARAMETER
                check_param = check_param.replace('_','')
                check_param = check_param.replace('GRO','')
                check_param = check_param.replace('QCHEM','')
                check_param = check_param.replace('XYZ','')
                check_param = check_param.replace('NDX','')
                if check_param != '':
                    raise configurationError(f"Invalid input for '{parameter_name}': {PARAMETER}")
            case 'CLUSTERS_TO_EXTRACT':
                if PARAMETER is None:
                    self.CONFIG.config[parameter_name] = 0
                if PARAMETER < -1:
                    raise configurationError(f"'{parameter_name}' outside valid range")
            case 'REACTANT_TO_PRINT':
                if PARAMETER is None or PARAMETER == []:
                    if self.CONFIG.config['CLUSTERS_TO_EXTRACT'] == -1 or self.CONFIG.config['CLUSTERS_TO_EXTRACT'] > 0:
                        if self.RANK == 0:
                            self.CONFIG.STDOUT.write('WARNING: No reactant state to extract defined, no clusters will be extracted\n')
                        self.CONFIG.config['CLUSTERS_TO_EXTRACT'] = 0
                    return
                if self.rankClustersExtracted():
                    self.CONFIG.config[parameter_name] = None
                    return
                if len(PARAMETER) > self.CONFIG.config['REACTION_SHELLS']:
                    raise configurationError(f"Shell number in '{parameter_name}' exceeds the requested reactant shell number")
                for _ in range(self.CONFIG.config['REACTION_SHELLS']-len(PARAMETER)):
                    PARAMETER.append({})
                for shell in PARAMETER:
                    if not isinstance(shell,dict):
                        raise configurationError(f"Non-dictionary shell content provided in '{parameter_name}': '{shell}', syntax is {SYNTAX}")
                    for residue,num in shell.items():
                        if residue not in self.CONFIG.config['RESIDUE_LIST']:
                            raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
                        if self.CONFIG.config['EXCLUDE_SOLVENT_RESIDUES'] is not None and residue in self.CONFIG.config['EXCLUDE_SOLVENT_RESIDUES']:
                            raise mySystemError(f"Invalid residue in '{parameter_name}', cannot contain residues excluded using 'EXCLUDE_SOLVENT_RESIDUES'")
                        if num < 0:
                            raise configurationError(f"Invalid residue number requested in {parameter_name} for '{residue}': {num}")
                if not self.CONFIG.configExists('CONSERVE_COORDINATION'):
                    for residue in self.CONFIG.config['RESIDUE_LIST']:
                        shell.setdefault(residue,0)
                self.REACTANT_TO_PRINT = self.GRAPH.rxnNode(configuration=self.CONFIG,solvation_shells=self.CONFIG.config['REACTANT_TO_PRINT'])
                #if self.RANK == 0:
                    #self.CONFIG.STDOUT.write(f"REACTANT TO PRINT: {self.REACTANT_TO_PRINT}\n")
            case 'CONSERVE_COORDINATION':
                if PARAMETER is None or PARAMETER == {}:
                    return
                all_none = True
                for residue,conserve_dict in PARAMETER.items():
                    if residue not in self.CONFIG.config['RESIDUE_LIST']:
                        raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
                    if conserve_dict is None:
                        continue
                    all_none = False
                    if not isinstance(conserve_dict,dict):
                        raise configurationError(f"Non-dictionary '{parameter_name}' parameter encountered for '{residue}': '{conserve_dict}'")
                    for conserve_residue,number in conserve_dict.items():
                        if number < 0:
                            raise configurationError(f"Invalid conserved residue number encountered in '{parameter_name}' for '{residue}': '{number}'")
                        if conserve_residue not in self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']:
                            raise configurationError(f"Non-'COARSEN_SOLVENT_RESIDUES' conserved residue encountered in '{parameter_name}' for '{residue}': '{conserve_residue}'")
                if all_none:
                    self.CONFIG.STDOUT.write("WARNING: 'CONSERVE_COORDINATION' contains no coodination conservation instructions.\n")
                    self.CONFIG.config[parameter_name] = None
            case 'COARSEN_SOLVENT_RESIDUES':
                if PARAMETER is None or PARAMETER == []:
                    return
                for residue in PARAMETER:
                    if residue not in self.CONFIG.config['RESIDUE_LIST']:
                        raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
            case 'EXCLUDE_SOLVENT_RESIDUES':
                if PARAMETER is None or PARAMETER == []:
                    return
                if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                    self.CONFIG.STDOUT.write(f"WARNING: '{parameter_name}' cannot be used with 'CONSERVE_COORDINATION', clearing. To exclude specific solvent types, ensure these types do not appear in 'CONSERVE_COORDINATION'.\n")
                    self.CONFIG.config[parameter_name] = None
                    return
                for residue in PARAMETER:
                    if residue not in self.CONFIG.config['RESIDUE_LIST']:
                        raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
                set_exclude = set(PARAMETER)
                set_coarsen = set(self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']) if self.CONFIG.config['COARSEN_SOLVENT_RESIDUES'] is not None else set()
                if not (set_exclude < set_coarsen):
                    self.CONFIG.config[parameter_name] = list(set_exclude & set_coarsen)
                    raise mySystemError(f"WARNING: Some excluded residues in {parameter_name} are not contained within 'COARSEN_SOLVENT_RESIDUES'. Correcting...")
                self.CONFIG.config[parameter_name] = list(set_exclude)
            case 'FILTER_REACTANTS_BY_Z':
                if PARAMETER is None:
                    return
                if isinstance(PARAMETER, tuple):
                    PARAMETER = list(PARAMETER)
                    self.CONFIG.config[parameter_name]
                if len(PARAMETER) % 2 == 1:
                    raise mySystemError(f"'{parameter_name}' does not include even number of bounds")
                for index in range(len(PARAMETER)-2):
                    if PARAMETER[index] > PARAMETER[index+1]:
                        raise mySystemError(f"'{parameter_name}' bounds are out of order, syntax is: {self.CONFIG.syntax[parameter_name]}")
                paired_list = []
                for index in range(0,len(PARAMETER),2):
                    paired_list.append((PARAMETER[index],PARAMETER[index+1]))
                PARAMETER = paired_list
                self.CONFIG.config[parameter_name] = PARAMETER
            case 'HBOND_TARGET':
                if PARAMETER is None:
                    return
                if PARAMETER <= 0:
                    raise configurationError(f"'{parameter_name}' outside valid range")
            case 'HBOND_DEV':
                if PARAMETER is None and self.CONFIG.config['HBOND_TARGET'] is None:
                    return
                if PARAMETER == None:
                    self.CONFIG.config[parameter_name] = 0
                    PARAMETER = self.CONFIG.config[parameter_name]
                else:
                    if PARAMETER <= 0:
                        raise configurationError(f"'{parameter_name}' outside valid range")
                    if self.CONFIG.config['HBOND_TARGET'] <= PARAMETER:
                        raise configurationError(f"'HBOND_TARGET' and '{parameter_name}' provided produce invalid H-Bond range")
            case 'SPECTATOR_TARGET':
                if PARAMETER is None:
                    return
                if (PARAMETER.keys() - self.CONFIG.config['RESIDUE_LIST'].keys()):
                    raise configurationError(f"'{parameter_name}' contains invalid residues: {', '.join([invalid_key for invalid_key in (PARAMETER.keys() - self.CONFIG.config['RESIDUE_LIST'].keys())])}")
                for residue,target in PARAMETER.items():
                    if not isinstance(target,int) and not isinstance(target,float):
                        raise configurationError(f"Non-numerical entry in '{parameter_name}', '{residue}':{target}")
                    if target < 0:
                        raise configurationError(f"Invalid entry in '{parameter_name}', '{residue}':{target}")
            case 'SPECTATOR_DEV':
                if PARAMETER is None and self.CONFIG.config['SPECTATOR_TARGET'] is None:
                    return
                if PARAMETER == None:
                    self.CONFIG.config[parameter_name] = {res:0 for res in self.CONFIG.config['SPECTATOR_TARGET'].keys()}
                    PARAMETER = self.CONFIG.config[parameter_name]
                else:
                    if (PARAMETER.keys() - self.CONFIG.config['RESIDUE_LIST'].keys()):
                        raise configurationError(f"'{parameter_name}' contains invalid residues: {', '.join([invalid_key for invalid_key in (PARAMETER.keys() - self.CONFIG.config['RESIDUE_LIST'].keys())])}")
                    if PARAMETER.keys() != self.CONFIG.config['SPECTATOR_TARGET'].keys():
                        raise mySystemError(f"'{parameter_name}' and 'SPECTATOR_TARGET' keys do not match")
                    for residue,target in PARAMETER.items():
                        if not isinstance(target,int) and not isinstance(target,float):
                            raise configurationError(f"Non-numerical entry in '{parameter_name}', '{residue}':{target}")
                        if target < 0:
                            raise configurationError(f"Invalid entry in '{parameter_name}', '{residue}':{target}")
            case 'RESIDUE_DIELECTRICS':
                if PARAMETER is None or PARAMETER == {}:
                    if self.CONFIG.configExists('OUTPUT_TYPE') and 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                        raise configurationError(f"QCHEM output was requested but no '{parameter_name}' was given")
                    return
                if PARAMETER.keys() != self.CONFIG.config['RESIDUE_LIST'].keys():
                    raise configurationError(f"Incomplete '{parameter_name}'. Missing residues {set(self.CONFIG.config['RESIDUE_LIST'].keys()) - set(PARAMETER.keys())}")
                for residue,dielectric in PARAMETER.items():
                    if not isinstance(dielectric,int) and not isinstance(dielectric,float):
                        raise configurationError(f"Non-numerical entry in '{parameter_name}', '{residue}':{dielectric}")
                    if dielectric < 0:
                        raise configurationError(f"Invalid entry in '{parameter_name}', '{residue}':{dielectric}")
            case 'RESIDUE_CHARGES':
                if PARAMETER is None or PARAMETER == {}:
                    if self.CONFIG.configExists('OUTPUT_TYPE') and 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                        raise configurationError(f"QCHEM output was requested but no '{parameter_name}' was given")
                    return
                if PARAMETER.keys() != self.CONFIG.config['RESIDUE_LIST'].keys():
                    raise configurationError(f"Incomplete '{parameter_name}'. Missing residues {set(self.CONFIG.config['RESIDUE_LIST'].keys()) - set(PARAMETER.keys())}")
                for residue,charge in PARAMETER.items():
                    if not isinstance(charge,int) and not isinstance(charge,float):
                        raise configurationError(f"Invalid entry in '{parameter_name}', '{residue}':{charge}")
            case 'QC_BASIS':
                if PARAMETER is None:
                    if self.CONFIG.configExists('OUTPUT_TYPE') and 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                        raise configurationError(f"QCHEM output was requested but no '{parameter_name}' was given")
                    return
                if isinstance(PARAMETER,dict):
                    user_atom_types = self.CONFIG.config['ATOM_TYPE_LIST']
                    check_atom_types = sorted(list(PARAMETER.keys()))
                    if sorted(user_atom_types) != check_atom_types:
                        raise mySystemError(f"'{parameter_name}' is not complete. System atom type list is: {user_atom_types}")
                    element_symbol_basis_dict = {}
                    for atom_type,basis in PARAMETER.items():
                        element = self.getElementSymbol(atom_type)
                        if element not in element_symbol_basis_dict:
                            element_symbol_basis_dict[element] = basis
                            continue
                        if element_symbol_basis_dict[element] != basis:
                            raise mySystemError(f"Basis set mismatch found for atom type '{atom_type}', element '{element}'. Multiple basis sets for single elements (BASIS MIXED) are not supported. If this atom type naming does not represent this element, please rename the atom type")
                    self.CONFIG.config[parameter_name] = element_symbol_basis_dict
            case 'QC_BASIS2':
                if PARAMETER is None:
                    return
                if isinstance(PARAMETER,dict):
                    user_atom_types = self.CONFIG.config['ATOM_TYPE_LIST']
                    check_atom_types = sorted(list(PARAMETER.keys()))
                    if sorted(user_atom_types) != check_atom_types:
                        raise mySystemError(f"'{parameter_name}' is not complete. System atom type list is: {user_atom_types}")
                    element_symbol_basis_dict = {}
                    for atom_type,basis2 in PARAMETER.items():
                        element = self.getElementSymbol(atom_type)
                        if element not in element_symbol_basis_dict:
                            element_symbol_basis_dict[element] = basis2
                            continue
                        if element_symbol_basis_dict[element] != basis2:
                            raise mySystemError(f"Auxillary basis set mismatch found for atom type '{atom_type}', element '{element}'. Multiple aux basis sets for single elements (BASIS2 MIXED) are not supported. If this atom type naming does not represent this element, please rename the atom type")
                    self.CONFIG.config[parameter_name] = element_symbol_basis_dict
            case 'QC_PSEUDO':
                if PARAMETER is None:
                    return
                
                element_symbol_basis_dict = {}
                for atom_type,pseudo in PARAMETER.items():
                    if atom_type not in self.CONFIG.config['ATOM_TYPE_LIST']:
                        raise mySystemError(f"Invalid atom type in '{parameter_name}': '{atom_type}'")
                    element = self.getElementSymbol(atom_type)
                    if element not in element_symbol_basis_dict:
                        element_symbol_basis_dict[element] = pseudo
                        continue
                    if element_symbol_basis_dict[element] != pseudo:
                        raise mySystemError(f"Pseudopotential mismatch found for atom type '{atom_type}', element '{element}'. Multiple pseudopotentials may not be assigned to a single element. If this atom type naming does not represent this element, please rename the atom type")
                
                if isinstance(self.CONFIG.config['QC_BASIS'],str):
                    self.CONFIG.config['QC_BASIS'] = {atom_type:self.CONFIG.config['QC_BASIS'] for atom_type in self.CONFIG.config['ATOM_TYPE_LIST']}
                    for atom_type,pseudo in self.CONFIG.config['QC_PSEUDO'].items():
                        self.CONFIG.config['QC_BASIS'][atom_type] = pseudo
                if isinstance(self.CONFIG.config['QC_BASIS2'],str):
                    self.CONFIG.config['QC_BASIS2'] = {atom_type:self.CONFIG.config['QC_BASIS2'] for atom_type in self.CONFIG.config['ATOM_TYPE_LIST']}
                    for atom_type,pseudo in self.CONFIG.config['QC_PSEUDO'].items():
                        self.CONFIG.config['QC_BASIS2'][atom_type] = pseudo

                self.CONFIG.config[parameter_name] = element_symbol_basis_dict
            case 'QC_METHOD':
                if PARAMETER is None:
                    if self.CONFIG.configExists('OUTPUT_TYPE') and 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                        raise configurationError(f"QCHEM output was requested but no '{parameter_name}' was given")
                    return
            case 'QC_PCM_METHOD':
                if PARAMETER is None:
                    if self.CONFIG.configExists('OUTPUT_TYPE') and 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                        raise configurationError(f"QCHEM output was requested but no '{parameter_name}' was given")
            case 'PROFILE':
                if PARAMETER is None:
                    self.CONFIG.config[parameter_name] = False
            case 'DEBUG':
                if PARAMETER is None:
                    self.CONFIG.config[parameter_name] = False
            case 'COORD_NUM_RESIDUE':
                if PARAMETER is None:
                    return
                if isinstance(PARAMETER,str):
                    self.CONFIG.config[parameter_name] = [PARAMETER]
                    PARAMETER = self.CONFIG.config[parameter_name]
                for residue in PARAMETER:
                    if residue not in self.CONFIG.config['RESIDUE_LIST']:
                        raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
            case 'K_LIGAND':
                if PARAMETER is None:
                    return
                if isinstance(PARAMETER,str):
                    self.CONFIG.config[parameter_name] = [PARAMETER]
                    PARAMETER = self.CONFIG.config[parameter_name]
                for residue in PARAMETER:
                    if residue not in self.CONFIG.config['RESIDUE_LIST']:
                        raise configurationError(f"Invalid residue encountered in '{parameter_name}': {residue}")
            case 'K_LIGAND_CONC':
                if PARAMETER is None:
                    return
                if not self.CONFIG.configExists('K_LIGAND'):
                    raise mySystemError(f"'K_LIGAND' not found")
                if isinstance(PARAMETER,(int,float)):
                    self.CONFIG.config[parameter_name] = [PARAMETER]
                    PARAMETER = self.CONFIG.config[parameter_name]
                for conc in PARAMETER:
                    if not isinstance(conc,(int,float)):
                        raise mySystemError(f"Non-numerical concentration encountered in '{PARAMETER}'")

                if len(self.CONFIG.config['K_LIGAND']) != len(PARAMETER):
                    raise mySystemError(f"K_LIGAND and K_LIGAND_CONC lengths do not match")
            case 'VIS_SUB_GRAPH_SIZE':
                #Fix!!
                return
                if PARAMETER <= 0:
                    raise mySystemError(f"Invalid '{PARAMETER}', must be larger than 0")
            case _:
                raise configurationError(f'Parameter {parameter_name} does not have an internal check case, this is not user issue')
            
        self.calculateLocalPeakMemUse()

    def parseDataLAMMPS(self):
        self.line = self.DATA_FILE.readline()

        #Minimum required information to parse
        self.NUM_ATOMS = self.NUM_ATOM_TYPES = ''
        #Extra info parsed from LAMMPS data files specifically
        self.NUM_BONDS = self.NUM_BOND_TYPES = self.NUM_ANGLES = self.NUM_ANGLE_TYPES = self.NUM_DIHEDRALS = self.NUM_DIHEDRAL_TYPES = self.NUM_IMPROPERS = self.NUM_IMPROPER_TYPES = ""

        while "Atoms # full" not in self.line:
            if 'LAMMPS' in self.line: #skip header
                self.line = self.DATA_FILE.readline()

            if "atom" in self.line and self.NUM_ATOMS == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_ATOMS += i

                self.NUM_ATOMS = int(self.NUM_ATOMS)

            if "atom types" in self.line and self.NUM_ATOM_TYPES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_ATOM_TYPES += i

                self.NUM_ATOM_TYPES = int(self.NUM_ATOM_TYPES)

            if "bonds" in self.line and self.NUM_BONDS == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_BONDS += i

                self.NUM_BONDS = int(self.NUM_BONDS)

            if "bond types" in self.line and self.NUM_BOND_TYPES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_BOND_TYPES += i

                self.NUM_BOND_TYPES = int(self.NUM_BOND_TYPES)

            if "angles" in self.line and self.NUM_ANGLES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_ANGLES += i

                self.NUM_ANGLES = int(self.NUM_ANGLES)

            if "angle types" in self.line and self.NUM_ANGLE_TYPES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_ANGLE_TYPES += i

                self.NUM_ANGLE_TYPES = int(self.NUM_ANGLE_TYPES)

            if "dihedrals" in self.line and self.NUM_DIHEDRALS == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_DIHEDRALS += i

                self.NUM_DIHEDRALS = int(self.NUM_DIHEDRALS)

            if "dihedral types" in self.line and self.NUM_DIHEDRAL_TYPES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_DIHEDRAL_TYPES += i

                self.NUM_DIHEDRAL_TYPES = int(self.NUM_DIHEDRAL_TYPES)

            if "impropers" in self.line and self.NUM_IMPROPERS == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_IMPROPERS += i

                self.NUM_IMPROPERS = int(self.NUM_IMPROPERS)

            if "improper types" in self.line and self.NUM_IMPROPER_TYPES == "":
                for i in self.line:
                    if i.isdigit():
                        self.NUM_IMPROPER_TYPES += i

                self.NUM_IMPROPER_TYPES = int(self.NUM_IMPROPER_TYPES)

            if "xlo" in self.line and "xhi" in self.line:
                self.line = self.line.strip()
                self.line = self.line.split()
                xlo = float(self.line[0])
                xhi = float(self.line[1])

            if "ylo" in self.line and "yhi" in self.line:
                self.line = self.line.strip()
                self.line = self.line.split()
                ylo = float(self.line[0])
                yhi = float(self.line[1])

            if "zlo" in self.line and "zhi" in self.line:
                self.line = self.line.strip()
                self.line = self.line.split()
                zlo = float(self.line[0])
                zhi = float(self.line[1])
                    
            self.line = self.DATA_FILE.readline()

        self.line = self.DATA_FILE.readline()

        self.boxLengthX = xhi - xlo
        self.boxLengthY = yhi - ylo
        self.boxLengthZ = zhi - zlo
        
        self.ATOMS = np.empty(self.NUM_ATOMS,dtype=int)
        self.ATOMTYPES = self.NUM_ATOMS * [None]
        self.MOLTYPES = np.empty(self.NUM_ATOMS,dtype=int)
        self.CHARGES = np.empty(self.NUM_ATOMS,dtype=float)
        self.X = np.empty(self.NUM_ATOMS,dtype=float)
        self.Y = np.empty(self.NUM_ATOMS,dtype=float)
        self.Z = np.empty(self.NUM_ATOMS,dtype=float)
        self.boxX = np.empty(self.NUM_ATOMS,dtype=int)
        self.boxY = np.empty(self.NUM_ATOMS,dtype=int)
        self.boxZ = np.empty(self.NUM_ATOMS,dtype=int)
        self.ATOM_DIELECTRICS = np.ones(self.NUM_ATOMS,dtype=float)

        dataCount = 0
        checkDataFormat = 0

        while self.line == '\n':
            self.line = self.DATA_FILE.readline()

        while self.line != '\n':
            
            index = 0

            self.line = self.line.strip()
            self.line = self.line.split()

            if not checkDataFormat:
                for data in self.line:

                    try:
                        int(data)
                        dataCount += 1
                    except ValueError:
                        try:
                            float(data)
                            dataCount += 1
                        except ValueError:
                            pass
                
                checkDataFormat = 1

            if dataCount == 7:
                index = int(self.line[0])
                self.ATOMS[index-1] = int(self.line[0])
                self.MOLTYPES[index-1] = int(self.line[1])
                self.ATOMTYPES[index-1] = int(self.line[2])
                self.CHARGES[index-1] = float(self.line[3])
                self.X[index-1] = float(self.line[4])
                self.Y[index-1] = float(self.line[5])
                self.Z[index-1] = float(self.line[6])
            elif dataCount == 10:
                index = int(self.line[0])
                self.ATOMS[index-1] = int(self.line[0])
                self.MOLTYPES[index-1] = int(self.line[1])
                self.ATOMTYPES[index-1] = int(self.line[2])
                self.CHARGES[index-1] = float(self.line[3])
                self.X[index-1] = float(self.line[4])
                self.Y[index-1] = float(self.line[5])
                self.Z[index-1] = float(self.line[6])
                self.boxX[index-1] = int(self.line[7])
                self.boxY[index-1] = int(self.line[8])
                self.boxZ[index-1] = int(self.line[9])
            else:
                self.CONFIG.STDOUT.write("Data file does not use 'full' atom style- using default parsing method. This may result in errors.\n")

                index = int(self.line[0])
                for data in range(dataCount):
                    if not self.ATOMS[index-1]:
                        self.ATOMS[index-1] = int(self.line[0])
                    elif not self.MOLTYPES[index-1]:
                        self.MOLTYPES[index-1] = int(self.line[1])
                    elif not self.ATOMTYPES[index-1]:
                        self.ATOMTYPES[index-1] = int(self.line[2])
                    elif not self.CHARGES[index-1]:
                        self.CHARGES[index-1] = float(self.line[3])
                    elif not self.X[index-1]:
                        self.X[index-1] = float(self.line[4])
                    elif not self.Y[index-1]:
                        self.Y[index-1] = float(self.line[5])
                    elif not self.Z[index-1]:
                        self.Z[index-1] = float(self.line[6])
                    elif not self.boxX[index-1]:
                        self.boxX[index-1] = int(self.line[7])
                    elif not self.boxY[index-1]:
                        self.boxY[index-1] = int(self.line[8])
                    elif not self.boxZ[index-1]:
                        self.boxZ[index-1] = int(self.line[9])

            self.line = self.DATA_FILE.readline()

        self.checkConfiguration('ATOM_TYPE_LIST')

        if self.RANK == 0:
            self.CONFIG.STDOUT.write("Data File Parsed\n")

        self.calculateLocalPeakMemUse()

    def connectivityLAMMPS(self):

        self.CONNECTIVITY = np.full(self.NUM_ATOMS,-1,dtype=np.int32) #Numpy array that stores (ordered or unordered) molecular connectivity via molecule indexes stored under atomic indexes (ex. [33,33,33,4,4,16,16,16,...])

        while "Bonds" not in self.line:
            self.line = self.DATA_FILE.readline()

        if not self.line:
            raise mySystemError("'Bonds' section missing from LAMMPS data file")
        
        self.line = self.DATA_FILE.readline() #Move past "Bonds" line in LAMMPSTRJ

        while self.line == '\n': #Move past any white space
            self.line = self.DATA_FILE.readline()

        molnum = 0
        molatoms = 0

        while self.line != '\n': #While in "Bonds" section

            self.line = self.line.strip()
            self.line = self.line.split()

            if len(self.line) == 4:
            
                #self.BONDTYPES[int(self.line[0])-1] == int(self.line[1])

                atomi = int(self.line[2])-1
                atomj = int(self.line[3])-1

                if self.CONNECTIVITY[atomi] == -1 and self.CONNECTIVITY[atomj] == -1: #Atom i and j have not been assigned to a molecule yet
                    self.CONNECTIVITY[atomi] = molnum
                    self.CONNECTIVITY[atomj] = molnum

                    molnum += 1
                    molatoms += 2
                elif self.CONNECTIVITY[atomj] == -1: #Only atom i has a molecule --> assign j to same molecule
                    self.CONNECTIVITY[atomj] = self.CONNECTIVITY[atomi]
                    molatoms += 1
                elif self.CONNECTIVITY[atomi] == -1: #Only atom j has a molecule --> assign i to same molecule
                    self.CONNECTIVITY[atomi] = self.CONNECTIVITY[atomj]
                    molatoms += 1
                elif self.CONNECTIVITY[atomi] < self.CONNECTIVITY[atomj]: #A bond has been found between two 'different' molecules meaning they are the same molecule and should be reassigned to the lowest molecule #
                    for atom in self.CONNECTIVITY:
                        if atom == self.CONNECTIVITY[atomj]:
                            atom = self.CONNECTIVITY[atomi]
                elif self.CONNECTIVITY[atomi] > self.CONNECTIVITY[atomj]: # " "
                    for atom in self.CONNECTIVITY:
                        if atom == self.CONNECTIVITY[atomi]:
                            atom = self.CONNECTIVITY[atomj]
                #else ... a duplicate bond entry has been found- atomi and atomj exist, atomi == atomj

                self.line = self.DATA_FILE.readline()
            else:
                self.CONFIG.STDOUT.write('Unable to parse "Bonds" section as written.\n')
                break

        #Create empty MOLECULE array, this allows for quick indexing using the molecule indexs contained in CONNECTIVITY
        #atomicspecies = sum(CONNECTIVITY == -1)
        single_atom_mask = self.CONNECTIVITY == -1

        num_single_atoms = np.sum(single_atom_mask)

        new_mol_seq = np.arange(molnum, molnum + num_single_atoms)

        self.CONNECTIVITY[single_atom_mask] = new_mol_seq
        molnum += num_single_atoms

        self.createMolecules()

        self.calculateLocalPeakMemUse()

    def splitGROMACSDataLine(self,line):

        #Splits line of GROMACS data (.gro) file into components based on line spacing rules for GROMACS data files
        #res number - cols 1-5
        #res string - cols 6-10
        #atom string - cols 11-15
        #atom number - cols 16-20
        #X - cols 21-28
        #Y - cols 29-36
        #Z = cols 37-44

        try:

            resNum = int(line[0:5].strip())
            resType = line[5:10].strip()
            atomType = line[10:15].strip()
            atomNum = int(line[15:20].strip())
            #must convert xyz from nm to angstrom
            X = float(line[20:28].strip()) * 10
            Y = float(line[28:36].strip()) * 10
            Z = float(line[36:44].strip()) * 10

            return (atomNum, atomType, resNum, resType, X, Y, Z)
        
        except IndexError: 
            return None
        except ValueError:
            return None

    def parseDataAndConnectivityGROMACS(self):

        #Parsing function for GROMACS .gro file

        #Minimum required information to parse
        self.NUM_ATOMS = self.NUM_ATOM_TYPES = None

        #Skip header
        self.line = self.DATA_FILE.readline()
        self.line = self.DATA_FILE.readline()

        #Store atom number line and create connectivity array
        self.NUM_ATOMS = int(self.line.strip())
        self.CONNECTIVITY = np.full(self.NUM_ATOMS,-1,dtype=np.int32) #Numpy array that stores (ordered or unordered) molecular connectivity via molecule indexes stored under atomic indexes (ex. [33,33,33,4,4,16,16,16,...])

        #Initialize atom data and XYZ arrays
        self.ATOMS = np.empty(self.NUM_ATOMS,dtype=int)
        self.ATOMTYPES = np.empty(self.NUM_ATOMS,dtype=object)
        self.X = np.empty(self.NUM_ATOMS,dtype=float)
        self.Y = np.empty(self.NUM_ATOMS,dtype=float)
        self.Z = np.empty(self.NUM_ATOMS,dtype=float)
        self.ATOM_DIELECTRICS = np.ones(self.NUM_ATOMS,dtype=float)
        residue_types = np.empty(self.NUM_ATOMS,dtype=object)
        temp_atom_type_set = set()
        carry_over = 0 #carry_over * 10^5 - Added to current atom number when it repeats at carry_over * 10^5 + 99999 + 1 (e.g. carry_over = 0, atom number repeats from 0 at 0*10^5 + 99999 + 1)

        self.line = self.DATA_FILE.readline()
        data = self.splitGROMACSDataLine(self.line)

        #Parse rest of data file
        while data != None:

            index = (carry_over * 100000) + data[0] - 1

            if data[0] == 99999:
                carry_over += 1

            self.ATOMS[index] = data[0]
            self.ATOMTYPES[index] = data[1]
            self.CONNECTIVITY[index] = data[2]-1
            residue_types[index] = data[3]

            self.X[index] = data[4]
            self.Y[index] = data[5]
            self.Z[index] = data[6]

            temp_atom_type_set.add(data[1])
            
            self.line = self.DATA_FILE.readline()
            data = self.splitGROMACSDataLine(self.line)

        #Store last line of .gro file containing box info
        self.line = self.line.strip().split()
        self.boxLengthX = float(self.line[0]) * 10
        self.boxLengthY = float(self.line[1]) * 10
        self.boxLengthZ = float(self.line[2]) * 10

        #Create molecules array from CONNECTIVITY
        self.createMolecules()

        for i in range(len(self.CONNECTIVITY)):
            if not self.MOLECULES[self.CONNECTIVITY[i]].residue:
                self.MOLECULES[self.CONNECTIVITY[i]].residue = residue_types[i]

        self.NUM_ATOM_TYPES = len(temp_atom_type_set)

        self.checkConfiguration('ATOM_TYPE_LIST',list(temp_atom_type_set))

        if self.RANK == 0:
            self.CONFIG.STDOUT.write("Data File Parsed\n")

        self.calculateLocalPeakMemUse()

    def createResidueList(self):
        #Creates a residue:atom_types reference dictionary from an array of molecules which have already been assigned residues, helpful for gromacs

        residue_list = {}

        if self.MOLECULES is None:
            return None
        
        for mol in self.MOLECULES:
            if not mol.residue:
                raise mySystemError('Encountered missing molecule residue while creating residue list')
            elif mol.residue in residue_list:
                if sorted(residue_list[mol.residue]) != sorted(self.ATOMTYPES[mol.atoms]):
                    raise mySystemError('Residue with arbitrary atom composition encountered')
                continue
            
            residue_list[mol.residue] = self.ATOMTYPES[mol.atoms]

        self.calculateLocalPeakMemUse()

        return residue_list

    def assignResidues(self):
        if self.MOLECULES is None:
            raise mySystemError('No molecules found when creating internal residue list')
        
        if not self.CONFIG.configExists('RESIDUE_LIST'):
            raise mySystemError('Residue dictionary is empty')
        
        #Iterate through molecule list and assign residues
        for mol in self.MOLECULES: 
            mol.residue = None 
            for res in self.CONFIG.config['RESIDUE_LIST']:
                if len(mol.atoms) != len(self.CONFIG.config['RESIDUE_LIST'][res]):
                    continue
                
                check_res = []
                for i in self.CONFIG.config['RESIDUE_LIST'][res]:
                    check_res.append(i)

                for atom in mol.atoms:
                    for res_atom in range(len(check_res)):
                
                        if self.ATOMTYPES[atom] == check_res[res_atom]:
                            check_res.pop(res_atom)
                            break

                if not check_res:
                    mol.residue = res
                    break

            if mol.residue is None:
                raise mySystemError('Molecule type encountered which is not in residue dictionary, check that RESIDUE_LIST is complete')
            
        self.calculateLocalPeakMemUse()

    def createMolecules(self):
        
        if np.any(self.CONNECTIVITY == -1):
            self.CONFIG.STDOUT.write('ERROR: Incomplete connectivity array provided to createMolecules()\n')
            return
        else:
            self.NUM_MOLECULES = np.unique(self.CONNECTIVITY).size

            self.MOLECULES = [None] * self.NUM_MOLECULES

            for i in range(self.CONNECTIVITY.size):

                if not self.MOLECULES[self.CONNECTIVITY[i]]:
                    self.MOLECULES[self.CONNECTIVITY[i]] = self.molecule(self.CONFIG,i,index=self.CONNECTIVITY[i])
                else:
                    self.MOLECULES[self.CONNECTIVITY[i]].addAtom(i)

        self.calculateLocalPeakMemUse()

    def createReactantList(self):

        #Create individual reactant lists for each process
        Unsplit_REACTANTS = []

        for molecule_number in range(len(self.MOLECULES)):
            if self.MOLECULES[molecule_number].residue == self.CONFIG.config['REACTANT']:
                Unsplit_REACTANTS.append(molecule_number)

        start_indexes,stop_indexes = self.splitIndexes(len(Unsplit_REACTANTS),self.NP)

        self.REACTANTS = np.array(Unsplit_REACTANTS[start_indexes[self.RANK]:stop_indexes[self.RANK]],dtype=np.int32)
        self.NUM_REACTANTS = self.REACTANTS.size

    def splitIndexes(self,arrayLength,numBins):

        #Takes in the length of the array to be split and returns two parallel arrays of length 'numBins' containing the start and stop index, respectively, for each bin

        BIN = arrayLength // numBins

        REM = arrayLength - (BIN * numBins)

        startIndexes = [None] * numBins
        stopIndexes = [None] * numBins

        for Bin in range(numBins):

            if Bin > 0:
                startIndexes[Bin] = stopIndexes[Bin-1]
            else:
                startIndexes[Bin] = 0

            if Bin == numBins-1: #on last available bin
                stopIndexes[Bin] = arrayLength
            elif REM:
                stopIndexes[Bin] = startIndexes[Bin] + BIN + 1
                REM -= 1
            else:
                stopIndexes[Bin] = startIndexes[Bin] + BIN

        return startIndexes,stopIndexes

    def arrangeBuffer(self,sendBuffer,sendDataType,ALL=True,optionalRecieveBuffer=None):

        #Only works for 1 dimensional numpy array sendBuffer object

        try: 
            assert isinstance(sendBuffer,np.ndarray) and sendBuffer.ndim == 1, "ERROR: Non 1DArray object passed into arrangeBuffer"

            sendcount = sendBuffer.size

            #sendDataType = sendBuffer.dtype

            #Gather total buffer size to gather into on all processes
            sendcount = np.int32(sendcount)
            all_counts = np.empty(self.NP,dtype=np.int32)
            self.COMM.Allgather([sendcount,MPI.INT],[all_counts,MPI.INT])

            #Based on individual send buffer counts create displacements corresponding to where each processes send buffer will be placed in the recieve buffer
            displacements = np.insert(np.cumsum(all_counts[:-1], dtype=np.int32), 0, 0)

            mpi_datatype = MPI._typedict[sendDataType.char]
            #mpi_datatype = dtlib.from_numpy_dtype(sendDataType)
            #mpi_datatype.Commit()

            if optionalRecieveBuffer is not None:
                try:
                    assert optionalRecieveBuffer.size == sum(all_counts), "ERROR: User recieve buffer incorrect size"

                    if ALL:
                        #Optional check to make sure that the buffers being combined all match in data type
                        # all_dtype_chars = self.COMM.allgather(sendBuffer.dtype.str)
                        # if len(set(all_dtype_chars)) != 1:
                        #     raise TypeError(
                        #         f"Inconsistent MPI buffer dtypes across ranks: "
                        #         f"{all_dtype_chars}"
                        #     )

                        self.COMM.Allgatherv([sendBuffer, sendcount, mpi_datatype], [optionalRecieveBuffer, all_counts, displacements, mpi_datatype])
                    else:
                        self.COMM.Gatherv([sendBuffer, sendcount, mpi_datatype], [optionalRecieveBuffer, all_counts, displacements, mpi_datatype])

                except AssertionError:
                    return
            else:
                optionalRecieveBuffer = np.empty(sum(all_counts),dtype=sendDataType)
                
                if ALL:
                    #Optional check to make sure that the buffers being combined all match in data type
                    # all_dtype_chars = self.COMM.allgather(sendBuffer.dtype.str)
                    # if len(set(all_dtype_chars)) != 1:
                    #     raise TypeError(
                    #         f"Inconsistent MPI buffer dtypes across ranks: "
                    #         f"{all_dtype_chars}"
                    #     )
                    self.COMM.Allgatherv([sendBuffer, sendcount, mpi_datatype], [optionalRecieveBuffer, all_counts, displacements, mpi_datatype])
                else:
                    self.COMM.Gatherv([sendBuffer, sendcount, mpi_datatype], [optionalRecieveBuffer, all_counts, displacements, mpi_datatype])

            return optionalRecieveBuffer
        except AssertionError:
            return

    def wrapMoleculesLAMMPS(self,start_index,stop_index):
        
        x_indexes = []
        x_changes = []
        y_indexes = []
        y_changes = []
        z_indexes = []
        z_changes = []

        #wrap molecules within MOLECULES_Seg (self.MOLECULES[i:f]) to prevent cutoffs at the periodic boundary
        for mol in self.MOLECULES[start_index:stop_index]:

            #Store molecule atoms
            mol_atoms = np.array(mol.atoms, dtype=np.int32)
        
            # Get most common box for X, Y, and Z
            most_common_x_box = Counter(self.boxX[mol_atoms]).most_common(1)[0][0]
            most_common_y_box = Counter(self.boxY[mol_atoms]).most_common(1)[0][0]
            most_common_z_box = Counter(self.boxZ[mol_atoms]).most_common(1)[0][0]
            
            # Create masks to find atoms that need to be wrapped
            x_mask = self.boxX[mol_atoms] != most_common_x_box
            y_mask = self.boxY[mol_atoms] != most_common_y_box
            z_mask = self.boxZ[mol_atoms] != most_common_z_box
            
            # Apply the masks to get the indexes and calculate the new positions
            if np.sum(x_mask) > 0:
                x_indexes.extend(mol_atoms[x_mask])
                x_changes.extend(self.X[mol_atoms[x_mask]] + (self.boxLengthX * (self.boxX[mol_atoms[x_mask]] - most_common_x_box)))

            if np.sum(y_mask) > 0:
                y_indexes.extend(mol_atoms[y_mask])
                y_changes.extend(self.Y[mol_atoms[y_mask]] + (self.boxLengthY * (self.boxY[mol_atoms[y_mask]] - most_common_y_box)))

            if np.sum(z_mask) > 0:
                z_indexes.extend(mol_atoms[z_mask])
                z_changes.extend(self.Z[mol_atoms[z_mask]] + (self.boxLengthZ * (self.boxZ[mol_atoms[z_mask]] - most_common_z_box)))

        self.calculateLocalPeakMemUse()

        return ( np.array(x_indexes), np.array(x_changes), np.array(y_indexes), np.array(y_changes), np.array(z_indexes), np.array(z_changes))

    def wrapMoleculesWrapperLAMMPS(self):
        
        if self.NP == 1: #Operating serially
            
            #self.CONFIG.STDOUT.write(f'RANK {self.RANK}: calling wrapMoleculesLAMMPS(self.MOLECULES[{startIndex}:{stopIndex}])\n')

            all_x_index,all_x_change,all_y_index,all_y_change,all_z_index,all_z_change = self.wrapMoleculesLAMMPS(0,self.NUM_MOLECULES)

        else:
            start,stop = self.splitIndexes(self.NUM_MOLECULES,self.NP)

            x_index,x_change,y_index,y_change,z_index,z_change = self.wrapMoleculesLAMMPS(start[self.RANK],stop[self.RANK])

            all_x_index = self.arrangeBuffer(x_index,np.dtype('int32'))
            all_x_change = self.arrangeBuffer(x_change,np.dtype('float64'))
            all_y_index = self.arrangeBuffer(y_index,np.dtype('int32'))
            all_y_change = self.arrangeBuffer(y_change,np.dtype('float64'))
            all_z_index = self.arrangeBuffer(z_index,np.dtype('int32'))
            all_z_change = self.arrangeBuffer(z_change,np.dtype('float64'))

            self.COMM.Barrier()
        
        for index, change in zip(all_x_index,all_x_change):
            self.X[index] = change
        for index,change in zip(all_y_index,all_y_change):
            self.Y[index] = change
        for index,change in zip(all_z_index,all_z_change):
            self.Z[index] = change

        self.calculateLocalPeakMemUse()

    def makePointParticles(self,start_index,stop_index):

        #Create point particle for each molecule from average XYZs:

        segment = stop_index - start_index

        #Pre-allocate the final, local interleaved 1D array for x,y and z
        local_ALL = np.empty(segment * 3, dtype=np.float64)

        #Define sliced views for X, Y, Z
        X_view = local_ALL[0::3]
        Y_view = local_ALL[1::3]
        Z_view = local_ALL[2::3]

        #Write point particle XYZ values into XYZ views:
        for offset, molecule_index in enumerate(range(start_index,stop_index)):
            molecule_atoms = self.MOLECULES[molecule_index].atoms

            X_view[offset] = np.mean(self.X[molecule_atoms])
            Y_view[offset] = np.mean(self.Y[molecule_atoms])
            Z_view[offset] = np.mean(self.Z[molecule_atoms])

        self.calculateLocalPeakMemUse()
        
        return local_ALL

    def createReactantClusters(self):

        #Pick self.CONFIG.config['CLUSTER_MOLECULES']-1 closest molecules to reactant

        for reactant in self.REACTANTS:

            radialDistanceList = self.allPBCDistance(self.POINTPARTICLESX[reactant],self.POINTPARTICLESY[reactant],self.POINTPARTICLESZ[reactant],self.POINTPARTICLESX,self.POINTPARTICLESY,self.POINTPARTICLESZ)

            # Sort the indices of the radial distance list to find the closest neighbors
            closest_neighbors = np.argsort(radialDistanceList)

            #Assign reactant clusters as X closest neighbors
            self.MOLECULES[reactant].cluster = closest_neighbors[:self.CONFIG.config['CLUSTER_MOLECULES']]

        self.calculateLocalPeakMemUse()

    def reactantClustersWrapper(self):

        if self.NP == 1:
            self.ALL_POINTPARTICLESXYZ = self.makePointParticles(0,self.NUM_MOLECULES)
        else:
            start,stop = self.splitIndexes(self.NUM_MOLECULES,self.NP)
            
            local_POINTPARTICLESXYZ = self.makePointParticles(start[self.RANK],stop[self.RANK])

            self.ALL_POINTPARTICLESXYZ = self.arrangeBuffer(local_POINTPARTICLESXYZ,np.dtype('float64'),optionalRecieveBuffer=self.ALL_POINTPARTICLESXYZ)

        self.POINTPARTICLESX = self.ALL_POINTPARTICLESXYZ[0::3]
        self.POINTPARTICLESY = self.ALL_POINTPARTICLESXYZ[1::3]
        self.POINTPARTICLESZ = self.ALL_POINTPARTICLESXYZ[2::3]

        self.createReactantClusters()

        self.calculateLocalPeakMemUse()

    def rankClustersExtracted(self):
        return (self.RANK_CLUSTERS_WRITTEN >= self.RANK_CLUSTERS_TO_EXTRACT) and (self.RANK_CLUSTERS_TO_EXTRACT != -1)

    def allRankClustersExtracted(self):
        return (self.TOTAL_CLUSTERS_WRITTEN >= self.TOTAL_CLUSTERS_TO_EXTRACT) and (self.TOTAL_CLUSTERS_TO_EXTRACT != -1)

    def gatherClusterExtractionStats(self):                                                                                                    
        #Creates send and recieve buffer space on each process for the cluster extraction counts, distributes the counts, and updates the total clusters written from the sum of the counts
        sendcount = np.int32([self.RANK_CLUSTERS_WRITTEN, 
                              self.REACTANTS_TO_WRITE_FOUND, 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_OVER_EXTRACT_FREQ'], 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_INNER_EXCLUDE_RES'], 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_INNER_NONCONSERVE'], 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONCONSERVE'], 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONSOLVENT'], 
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_SECONDOUTER_UNDERCOORD'],
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_OVER_HBOND_COUNT'],
                              self.EXTRACT_FAIL_STATS['WRITE_FAILED_UNDER_HBOND_COUNT']]) #Number of clusters extracted as a 32-bit integer
        send_stat_num = sendcount.size
        all_counts = np.empty(self.NP*send_stat_num,dtype=np.int32) #Numpy array initialized to hold the 32-bit int cluster count from each process (only on RANK ==0)
        self.COMM.Allgather([sendcount,MPI.INT],[all_counts,MPI.INT]) #Gather the cluster counts from each process and store them in all_counts, this is a synchronous process
        self.TOTAL_CLUSTERS_WRITTEN = sum(all_counts[::send_stat_num]) #Total clusters extracted = sum of all individual extraction counts
        self.TOTAL_REACTANTS_TO_WRITE_FOUND = sum(all_counts[1::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_OVER_EXTRACT_FREQ'] = sum(all_counts[2::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_INNER_EXCLUDE_RES'] = sum(all_counts[3::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_INNER_NONCONSERVE'] = sum(all_counts[4::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_FIRSTOUTER_NONCONSERVE'] = sum(all_counts[5::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_FIRSTOUTER_NONSOLVENT'] = sum(all_counts[6::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_SECONDOUTER_UNDERCOORD'] = sum(all_counts[7::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_OVER_HBOND_COUNT'] = sum(all_counts[8::send_stat_num])
        self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_UNDER_HBOND_COUNT'] = sum(all_counts[9::send_stat_num])

    def makeElementSymbolList(self):
        if not self.CONFIG.configExists('OUTPUT_TYPE') or ('XYZ' not in self.CONFIG.config['OUTPUT_TYPE'] and 'QCHEM' not in self.CONFIG.config['OUTPUT_TYPE']): #If no output file that uses element symbol labeling is requested simply use user atom type labels
            self.ELEMENTSYMBOLS = self.ATOMTYPES
            return
        
        self.ELEMENTSYMBOLS = np.empty(self.NUM_ATOMS,dtype=object)
        if self.ATOMTYPES.size != self.ELEMENTSYMBOLS.size:
            raise mySystemError('Atom type array size does not match element symbol array size')
        atomtypes = np.unique(self.ATOMTYPES)
        type_indexes = [self.ATOMTYPES == type for type in atomtypes]
        for type,indexes in zip(atomtypes,type_indexes):
            if type in self.ELEMENTS:
                self.ELEMENTSYMBOLS[indexes] = type
                continue
            if len(type) == 1:
                raise mySystemError(f"Unable to convert atom-type to element symbol: '{type}'. Please provide 'ATOM_TYPE_LIST'")
            if type[:2] in self.ELEMENTS:
                self.ELEMENTSYMBOLS[indexes] = type[:2]
                continue
            if type[:1] in self.ELEMENTS:
                self.ELEMENTSYMBOLS[indexes] = type[:1]
                continue

            type_as_string = type.item()
            if type_as_string[1].isupper():
                type_as_string[1] = type_as_string[1].lower()
                if type_as_string[:2] in self.ELEMENTS:
                    self.ELEMENTSYMBOLS[indexes] = type_as_string[:2]
                    continue

            raise mySystemError(f"Unable to convert atom-type to element symbol: '{type}'")
                                                                                                               
    def getElementSymbol(self,atom_type:str):
        if atom_type in self.ELEMENTS:
            return atom_type
        if len(atom_type) == 1:
            raise mySystemError(f"Unable to convert atom-type to element symbol: '{atom_type}'")
        if atom_type[:2] in self.ELEMENTS:
            return atom_type[:2]
        if atom_type[:1] in self.ELEMENTS:
            return atom_type[:1]
        raise mySystemError(f"Unable to convert atom-type to element symbol: '{atom_type}'")

    def extractReactantClusters(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        if self.rankClustersExtracted():
            return
        
        for reactant_index in self.REACTANTS:

            if self.rankClustersExtracted():
                break

            reactant = self.MOLECULES[reactant_index]

            if reactant.extract_shells is None:
                continue

            self.RANK_CLUSTERS_WRITTEN += 1

            if self.CONFIG.config['DEBUG']:
                self.debugCoordinationShells(reactant)

            if 'XYZ' in self.CONFIG.config['OUTPUT_TYPE']:
                self.extractClusterXYZFormat(reactant)
            if 'QCHEM' in self.CONFIG.config['OUTPUT_TYPE']:
                self.extractClusterQchemFormat(reactant)
            if 'GRO' in self.CONFIG.config['OUTPUT_TYPE']:
                self.extractClusterGroFormat(reactant)
            if 'NDX' in self.CONFIG.config['OUTPUT_TYPE']:
                self.extractClusterGMXIndexFormat(reactant)

        self.calculateLocalPeakMemUse()

    def clusterFileHeader(self,reactant):
        return f'Reactant cluster for frame: {self.STEP}, reactant index: {reactant.index}, H-Bonds: {reactant.extract_shells_hbonds}, outer shell charge: {reactant.outer_shell}\n'

    def extractClusterXYZFormat(self,reactant:molecule):

        outfile = open('./' + self.CONFIG.config['WRITE_DIRECTORY'] + '/' + self.CONFIG.config['RUN_NAME'] + '/' + self.CONFIG.config['RUN_NAME'] + "_" + str(self.RANK_CLUSTERS_WRITTEN+self.CLUSTER_WRITE_OFFSET) + ".xyz", "w")

        #----------------write header---------------------------------------------------------

        header = self.clusterFileHeader(reactant)

        #----------------write xyz portion--------------------------------------------------

        write = ''

        atom_num = 0

        for atom_index in reactant.atoms:
            atom_num += 1
            write += f"{self.ELEMENTSYMBOLS[atom_index]} {self.X[atom_index]} {self.Y[atom_index]} {self.Z[atom_index]}\n"

        refX = self.POINTPARTICLESX[reactant.index]
        refY = self.POINTPARTICLESY[reactant.index]
        refZ = self.POINTPARTICLESZ[reactant.index]

        for molecule_index in reactant.extract_shells:
            molecule = self.MOLECULES[molecule_index]
            for atom_index in molecule.atoms:
                atom_num += 1
                relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom_index],refX,self.Y[atom_index],refY,self.Z[atom_index],refZ,returnComponents=True)
                write += f"{self.ELEMENTSYMBOLS[atom_index]} {refX+relativeX} {refY+relativeY} {refZ+relativeZ}\n"

        outfile.write(f'{atom_num}\n{header}{write}')
        self.calculateLocalPeakMemUse()
        outfile.close()

    def extractClusterQchemFormat(self,reactant:molecule):

        #Determine atomic dielectric embedding functions within cluster
        self.reactantClusterDielectric(reactant)

        outfile = open('./' + self.CONFIG.config['WRITE_DIRECTORY'] + '/' + self.CONFIG.config['RUN_NAME'] + '/' + self.CONFIG.config['RUN_NAME'] + "_" + str(self.RANK_CLUSTERS_WRITTEN+self.CLUSTER_WRITE_OFFSET) + ".in", "w")

        #----------------write header---------------------------------------------------------
        
        header = f'!{self.clusterFileHeader(reactant)}'

        charge = 0
        for molecule_index in {reactant.index} | reactant.extract_shells:
            molecule = self.MOLECULES[molecule_index]
            res = molecule.residue
            charge += self.CONFIG.config['RESIDUE_CHARGES'][res]
        
        write = f"!------Electronic/Nuclear Configurations------\n$molecule\n{charge} 1\n" #Only writes singlet input files (maybe change)

        #----------------write xyz portion--------------------------------------------------

        atom_order = []
        elements = set()

        for atom_index in reactant.atoms:
            atom_order.append(atom_index)
            element_symbol = self.ELEMENTSYMBOLS[atom_index]
            elements.add(element_symbol)
            write += f"{element_symbol} {self.X[atom_index]} {self.Y[atom_index]} {self.Z[atom_index]}\n"

        refX = self.POINTPARTICLESX[reactant.index]
        refY = self.POINTPARTICLESY[reactant.index]
        refZ = self.POINTPARTICLESZ[reactant.index]

        for molecule_index in reactant.extract_shells:
            molecule = self.MOLECULES[molecule_index]
            for atom_index in molecule.atoms:
                atom_order.append(atom_index)
                element_symbol = self.ELEMENTSYMBOLS[atom_index]
                elements.add(element_symbol)
                relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom_index],refX,self.Y[atom_index],refY,self.Z[atom_index],refZ,returnComponents=True)
                write += f"{element_symbol} {refX+relativeX} {refY+relativeY} {refZ+relativeZ}\n"

        write += '$end\n\n'

        #----------------write job control sections (qchem only)-------------------------------
        optional_basis_section = ''
        optional_basis2_section = ''
        optional_ecp_section = ''
        write += '!------Calculation Type Section------\n$rem\n'
        write += f"METHOD\t{self.CONFIG.config['QC_METHOD']}\t\t\t\t!Method of approximating the schrodinger equation\n"
        if isinstance(self.CONFIG.config['QC_BASIS'],str):
            write += f"BASIS\t{self.CONFIG.config['QC_BASIS']}\t\t\t\t!Primary basis set\n"
        else:
            write += f"BASIS\tGEN\t\t\t\t!Primary basis set\n"
            optional_basis_section = '$basis\n' + '\n'.join([f"{element} 0\n{self.CONFIG.config['QC_BASIS'][element]}\n****" for element in elements]) + '\n$end\n\n'
        if self.CONFIG.configExists('QC_BASIS2'):
            if isinstance(self.CONFIG.config['QC_BASIS2'],str):
                write += f"BASIS2\t{self.CONFIG.config['QC_BASIS2']}\t\t\t\t!Auxilary basis set\n"
            else:
                write += f"BASIS2\tGEN\t\t\t\t!Auxilary basis set\n"
                optional_basis2_section = '$basis2\n' + '\n'.join([f"{element} 0\n{self.CONFIG.config['QC_BASIS2'][element]}\n****" for element in elements]) + '\n$end\n\n'
            write += f"DUAL_BASIS_ENERGY\tTRUE\t\t\t\t!Perform SCF energy calculation using dual-basis approximation\n"
        if self.CONFIG.configExists('QC_PSEUDO'):
            write += f"ECP\tGEN\t\t\t\t!Pseudopotential\n"
            optional_ecp_section = '$ecp\n' + '\n'.join([f"{element} 0\n{self.CONFIG.config['QC_PSEUDO'][element]}\n****" for element in elements if element in self.CONFIG.config['QC_PSEUDO']]) + '\n$end\n\n'
        
        write += 'JOB_TYPE SP\t\t\t\t!Single point calculation\n'
        write += 'SCF_ALGORITHM DIIS\t\t\t\t!SCF convergence algorithm\n'
        write += 'MEM_TOTAL 10000\t\t\t\t!Increase total memory (10 G)\n'
        
        write += 'SCF_CONVERGENCE 8\t\t\t\t!SCF convergence threshold, 10^-X\n'
        write += 'THRESH 12\t\t\t\t!Integral threshold, 10^-X (recommended to be >=(SCF_CONVERGENCE+3)\n'
        write += 'SCF_MAX_CYCLES 1000\t\t\t\t!Max convergence cycles before failure\n'
        write += 'SOLVENT_METHOD PCM\t\t\t\t!Enable Polarizable Continuum Solvation\n'

        write += 'MOLDEN_FORMAT 1\t\t\t\t!Enable molden usage\n'
        write += 'GUI 2\t\t\t\t!IQMOL compatable output style\n'
        write += 'PCM_PRINT 1\t\t\t\t!High verbosity to print atomic epsilon settings\n'
        write += '$end\n\n'

        write += optional_basis_section
        write += optional_basis2_section
        write += optional_ecp_section

        write += '$van_der_waals\n92\t1.86\t\t\t\t!Manually input vdw radii for uranium\n$end\n\n'

        write += '!------PCM Section------\n$pcm\n'
        write += 'het-pcm\t\t\t\t!Keyword to initialize Het-PCM\n'
        write += 'THEORY CPCM\t\t\t\t!Continuum model\n'
        write += '$end\n\n'

        write += '!------Atomic Dielectric Section------\n$atomic_epsilon\n'
        for index,atom_index in enumerate(atom_order):
            write += str(index+1) + ' ' + str(self.ATOM_DIELECTRICS[atom_index]) + '\n'
        write += '$end\n\n'

        outfile.write(f'{header}{write}')
        self.calculateLocalPeakMemUse()
        outfile.close()

    def extractClusterGroFormat(self,reactant:molecule):

        outfile = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/{self.CONFIG.config['RUN_NAME']}_{self.RANK_CLUSTERS_WRITTEN+self.CLUSTER_WRITE_OFFSET}.gro", "w")

        #***********
        # hbondfile = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/edges.txt", "w")
        # order_vs_index_atoms = []
        # order_vs_index_mols = []
        #***********

        header = self.clusterFileHeader(reactant)

        write = ''

        res_num = 1
        atom_num = 0

        #**************
        # order_vs_index_mols.append(reactant.index)
        #****************

        for atom_index in reactant.atoms:

            #*******
            # order_vs_index_atoms.append(atom_index)
            #*******

            if atom_num == 99999: #.gro files are 1 indexed and wrap elements when they reach the highest value their line container allows, e.g. 99999 for container length of 5  
                atom_num = 0
            else:
                atom_num += 1
            
            write += f"{res_num:>5}{reactant.residue:<5}{self.ATOMTYPES[atom_index]:>5}{atom_num:>5}{(self.X[atom_index]/10):>8.3f}{(self.Y[atom_index]/10):>8.3f}{(self.Z[atom_index]/10):>8.3f}\n"

        refX = self.POINTPARTICLESX[reactant.index]
        refY = self.POINTPARTICLESY[reactant.index]
        refZ = self.POINTPARTICLESZ[reactant.index]

        for molecule_index in reactant.extract_shells:

            molecule = self.MOLECULES[molecule_index]

            if res_num == 99999:
                res_num = 0
            else:
                res_num += 1

            #************
            # order_vs_index_mols.append(molecule_index)
            #***********

            for atom_index in self.MOLECULES[molecule_index].atoms:

                #*******
                # order_vs_index_atoms.append(atom_index)
                #*******

                relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom_index],refX,self.Y[atom_index],refY,self.Z[atom_index],refZ,returnComponents=True)
                X = refX + relativeX
                Y = refY + relativeY
                Z = refZ + relativeZ

                if atom_num == 99999:
                    atom_num = 0
                else:
                    atom_num += 1
                
                write += f"{res_num:>5}{molecule.residue:<5}{self.ATOMTYPES[atom_index]:>5}{atom_num:>5}{(X/10):>8.3f}{(Y/10):>8.3f}{(Z/10):>8.3f}\n"

        #Write box X,Y,Z at end of file
        write += f'{(self.boxLengthX/10):10.5f}{(self.boxLengthY/10):10.5f}{(self.boxLengthZ/10):10.5f}\n'

        header += f'{atom_num}\n'

        #*************
        # if (reactant.index == 748 and current_molecule_index == 2708) or (reactant.index == 748 and current_molecule_index == 874):
        #     pass

        # hbond_write = 'i mol_id_i j mol_id_j    //     mol_i_index i_index mol_j_index j_index\n'
        # total_extract_shells = {reactant.index} | reactant.extract_shells
        # for molecule_index in total_extract_shells:
        #     molecule = self.MOLECULES[molecule_index]
        #     for donor_atom_index,acceptor_list in molecule.atom_hbond_donation_atoms.items():
        #         for acceptor_mol_atom_tuple in acceptor_list:
        #             if acceptor_mol_atom_tuple[0] in total_extract_shells:
        #                 hbond_write += f'{order_vs_index_atoms.index(donor_atom_index)} {order_vs_index_mols.index(molecule_index)+1}  {order_vs_index_atoms.index(acceptor_mol_atom_tuple[1])} {order_vs_index_mols.index(acceptor_mol_atom_tuple[0])+1}\t\t{molecule_index} {donor_atom_index} {acceptor_mol_atom_tuple[0]} {acceptor_mol_atom_tuple[1]}\n'
        # hbondfile.write(hbond_write)
        #*************

        outfile.write(f'{header}{write}')
        self.calculateLocalPeakMemUse()
        outfile.close()

    def extractClusterGMXIndexFormat(self,reactant:molecule):
        #Creates .ndx file containing two atom group labels: [Cluster] - containing the atom indexes of the cluster and [Rest] - containing the atom indexes of the rests of the system
        #Atom indexes are 1 indexed via gromacs syntax
        #Provides first step in using gromacs to calculate the energy of the cluster as (cluster-cluster) + (cluster-rest)

        if self.GMX_SYSTEM_INDEXES is None:
            self.GMX_SYSTEM_INDEXES = {atom for atom in range(self.NUM_ATOMS)}
        outfile = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/{self.CONFIG.config['RUN_NAME']}_{self.RANK_CLUSTERS_WRITTEN+self.CLUSTER_WRITE_OFFSET}.ndx", "w")

        header = ';' + self.clusterFileHeader(reactant) + f";Trajectory file path: {self.CONFIG.config['TRJ_FILE_PATH']}\n"
        outfile.write(header)
        write = '[Cluster]\n'
        cluster = {atom for atom in reactant.atoms} | {atom for molecule in reactant.extract_shells for atom in self.MOLECULES[molecule].atoms}
        rest = self.GMX_SYSTEM_INDEXES - cluster

        character_count = 0
        for index,atom in enumerate(cluster):
            atom_string = str(atom+1) + ' '
            atom_string_length = len(atom_string)
            character_count += atom_string_length
            if character_count > 4094:
                outfile.write(write[:-1] + '\n')
                write = ''
                character_count = atom_string_length

            write += atom_string

            if index == len(cluster)-1:
                outfile.write(write[:-1] + '\n')                
        
        write = '[Rest]\n'

        character_count = 0
        for index,atom in enumerate(rest):
            atom_string = str(atom+1) + ' '
            atom_string_length = len(atom_string)
            character_count += atom_string_length
            if character_count > 4094:
                outfile.write(write[:-1] + '\n')
                write = ''
                character_count = atom_string_length

            write += atom_string

            if index == len(rest)-1:
                outfile.write(write[:-1] + '\n')

    #fix hbonding analysis for molecules other than water
    def hBonds(self):

        global clusterHBonds #This needs fixing, I believe this stores the same data as self.MOLECULES[self.REACTANTS].hBonds

        clusterHBonds = [0] * len(self.REACTANTS)
        hBondConnectivity = [[0 for x in range(self.NUM_ATOMS)]  for y in range(self.NUM_ATOMS)]

        #For each atom in cluster check for: 1) if atom is H    2) if H is bound to O, N, or F   3) if H is within 2.5 A of another O, N, or F   4) if H - (O,N,F) bond angle is within 145*-180*
        for i in range(len(self.REACTANTS)):
            for j in range(len(self.MOLECULES[self.REACTANTS[i]].cluster)):
                for k in range(len(self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[j]].atoms)):
                    if self.ATOMTYPES[self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[j]].atoms[k]] == 'H':
                        dipole = -1
                        H = self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[j]].atoms[k]
                        for l in range(len(self.BONDS[self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[j]].atoms[k]])):
                            if self.BONDS[self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[j]].atoms[k]][l] and (self.ATOMTYPES[l] == 'O' or self.ATOMTYPES[l] == 'o' or self.ATOMTYPES[l] == 'N' or self.ATOMTYPES[l] == 'n' or self.ATOMTYPES[l] == 'F' or self.ATOMTYPES[l] == 'f'): #check for possible hbond species
                                dipole = l
                                break
                        if dipole != -1:
                            for l in range(len(self.MOLECULES[self.REACTANTS[i]].cluster)):
                                if l != j: #filter for hbonds on the same molecule
                                    for m in range(len(self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[l]].atoms)):
                                        ONF = self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[l]].atoms[m]
                                        index = self.MOLECULES[self.MOLECULES[self.REACTANTS[i]].cluster[l]].atoms[m]
                                        if (self.ATOMTYPES[index] == 'O' or self.ATOMTYPES[index] == 'o' or self.ATOMTYPES[index] == 'N' or self.ATOMTYPES[index] == 'n' or self.ATOMTYPES[index] == 'F' or self.ATOMTYPES[index] == 'f'): #check for possible hbond species
                                            #at this point 'k' is our H, 'dipole' is our bonded O,N or F, and 'm' is our potential
                                            hBondLength = math.sqrt((self.X[H]-self.X[ONF])**2 + (self.Y[H]-self.Y[ONF])**2 + (self.Z[H]-self.Z[ONF])**2) #hydrogen bond radius (b)
                                            bondLength = math.sqrt((self.X[H]-self.X[dipole])**2 + (self.Y[H]-self.Y[dipole])**2 + (self.Z[H]-self.Z[dipole])**2) # H-(O,N,F) bond length (a)
                                            hypotenuse = math.sqrt((self.X[dipole]-self.X[ONF])**2 + (self.Y[dipole]-self.Y[ONF])**2 + (self.Z[dipole]-self.Z[ONF])**2) # nonbonced (O,N,F) - (O,N,F) length (c)
                                            if (hBondLength <= 2.5): #check for <2.5 H-(O,N,F) radius
                                                angle = math.degrees(math.acos((bondLength**2 + hBondLength**2 - hypotenuse**2) / (2 * bondLength * hBondLength))) #use law of cosines to find (O,N,F) - H  - (O,N,F) angle: angle = cos^-1((a^2 + b^2 - c^2)/(2*a*b))
                                                if (145 <= angle <= 180): #check for angle between 145 and 180
                                                    clusterHBonds[i] += 1
                                                    self.MOLECULES[self.REACTANTS[i]].hbonds += 1
                                                    hBondConnectivity[H][ONF] = 1
                                                    hBondConnectivity[ONF][H] = 1

        self.CONFIG.STDOUT.write("H-Bond Connectivity Analysis Complete\n")

    def readFrameLAMMPSOld(self):

        traj_atoms = None
        atom_num_changed = None
        id = None

        if not self.line:
            self.line = self.TRJ_FILE.readline()

        if 'TIMESTEP' in self.line:
            self.line = self.TRJ_FILE.readline()
            self.TIMESTEP = int(self.line.strip())
            while "ITEM: ATOMS" not in self.line:
                if "NUMBER OF ATOMS" in self.line:
                    self.line = self.TRJ_FILE.readline()
                    traj_atoms = self.line.strip()

                    traj_atoms = int(traj_atoms)

                    if traj_atoms != self.NUM_ATOMS:
                        atom_num_changed = 1

                if "BOX BOUNDS" in self.line:

                    self.line = self.TRJ_FILE.readline()
                    self.line = self.line.strip()
                    xlo,xhi = self.line.split()

                    self.boxLengthX = float(xhi) - float(xlo)

                    self.line = self.TRJ_FILE.readline()
                    self.line = self.line.strip()
                    ylo,yhi = self.line.split()

                    self.boxLengthY = float(yhi) - float(ylo)

                    self.line = self.TRJ_FILE.readline()
                    self.line = self.line.strip()
                    zlo,zhi = self.line.split()

                    self.boxLengthZ = float(zhi) - float(zlo)

                self.line = self.TRJ_FILE.readline()

            self.line = self.TRJ_FILE.readline()

            #skip whitespace
            while self.line == '\n':
                self.line = self.TRJ_FILE.readline()

        while "ITEM: TIMESTEP" not in self.line and self.line:
            self.line = self.line.strip()
            data = self.line.split()
            id = int(data[0])
            type = int(data[1])
            self.X[id-1] = float(data[2])
            self.Y[id-1] = float(data[3])
            self.Z[id-1] = float(data[4])
            self.boxX[id-1] = int(data[5])
            self.boxY[id-1] = int(data[6])
            self.boxZ[id-1] = int(data[7])

            self.line = self.TRJ_FILE.readline()

        if atom_num_changed:
            raise mySystemError("Atom number changed while reading trajectory",flush=True)

        if not self.line:
            self.PASS = False
            return

        self.calculateLocalPeakMemUse()

    def streamFramesLAMMPS(self,STOP,START=0,STRIDE=1):

        header_lines = 9 #Number of lines in LAMMPS frame header for ITEM:X blocks, atom number, and current box dimensions
        atom_type_full_length = 8 #Number of columns in LAMMPS atomtype=full output style
        frame_index = 0 #What frame index are we currently on?

        #Create memory map of LAMMPS trajectory
        mm = mmap.mmap(self.TRJ_FILE.fileno(), 0, access=mmap.ACCESS_READ)
        
        frame_start = 0
        atom_section_start = -1
        frame_end = -1
        file_size = mm.size()

        #Skip forward to the user-requested start frame
        while frame_index != START:
            frame_start = mm.find(b'ITEM: TIMESTEP', frame_start+1) #Look for the start of the next frame, starting at byte 1 to avoid catching the current 'ITEM: TIMESTEP'
            if frame_start == -1:
                raise mySystemError(f"START frame {START} exceeds total frames in file")
            frame_index += 1
        
        while frame_start < file_size and (frame_index < STOP or STOP == -1): #While not EOF or user requested STOP frame
            
            atom_section_start = frame_start
            for header_index in range(header_lines):
                #Trace byte location of end of atom header
                atom_section_start = mm.find(b'\n', atom_section_start) + 1
                #Using header_index end of line counter grab important data from header
                match header_index:
                    case 0:
                        timestep_start = atom_section_start
                    case 1:
                        self.TIMESTEP = int(mm[timestep_start:atom_section_start-1].decode('utf-8'))
                    case 2:
                        num_atoms_start = atom_section_start
                    case 3:
                        num_atoms = int(mm[num_atoms_start:atom_section_start-1].decode('utf-8'))
                        if self.NUM_ATOMS != num_atoms:
                            raise mySystemError(f'Incorrect atom number encountered in frame {self.STEP}')
                    case 4:
                        boxX_start = atom_section_start
                    case 5:
                        boxY_start = atom_section_start
                        X_arr = np.fromstring(mm[boxX_start:atom_section_start-1].decode('utf-8'), dtype=np.float64, sep=' ')
                        self.boxLengthX = X_arr[1] - X_arr[0]
                    case 6:
                        boxZ_start = atom_section_start
                        Y_arr = np.fromstring(mm[boxY_start:atom_section_start-1].decode('utf-8'), dtype=np.float64, sep=' ')
                        self.boxLengthY = Y_arr[1] - Y_arr[0] 
                    case 7:
                        Z_arr = np.fromstring(mm[boxZ_start:atom_section_start-1].decode('utf-8'), dtype=np.float64, sep=' ')
                        self.boxLengthZ = Z_arr[1] - Z_arr[0]
                    case 8:
                        pass
            
            #Trace byte location of end of trajectory section for frame
            frame_end = atom_section_start
            for _ in range(self.NUM_ATOMS):
                frame_end = mm.find(b'\n', frame_end) + 1
                if frame_end == 0:
                    raise mySystemError('Premature EOF encountered when reading LAMMPS trajectory, file may be truncated')
                
            if atom_section_start == frame_end:
                raise mySystemError(f'Unable to parse atom section of for frame: {self.STEP}')

            atom_bytes = mm[atom_section_start:frame_end]
                
            flat_atom_data = np.fromstring(atom_bytes.decode('utf-8'), dtype=np.float64, sep=' ')

            if flat_atom_data.size // atom_type_full_length != self.NUM_ATOMS:
                raise mySystemError(f'Inconsistant atom number encountered at frame {self.STEP}')
            
            #Reshape flat_atomic_data in id0,type0,x0,y0,z0,bx0,by0,bz0,id1,type1,x1,y1,z1,bx1,by1,bz1,... format into [[id0,id1,...],[type0,type1,...],...] format
            atomic_data_columated = flat_atom_data.reshape(-1, atom_type_full_length).T
            indexes = (atomic_data_columated[0].astype(np.int64)) - 1 #Convert LAMMPS id array into integer index array
            X_unsorted = atomic_data_columated[2]
            Y_unsorted = atomic_data_columated[3]
            Z_unsorted = atomic_data_columated[4]
            boxX_unsorted = atomic_data_columated[5]
            boxY_unsorted = atomic_data_columated[6]
            boxZ_unsorted = atomic_data_columated[7]

            self.X[indexes] = X_unsorted
            self.Y[indexes] = Y_unsorted
            self.Z[indexes] = Z_unsorted
            self.boxX[indexes] = boxX_unsorted
            self.boxY[indexes] = boxY_unsorted
            self.boxZ[indexes] = boxZ_unsorted

            #As soon as the current frame is processed yield the frame index
            yield frame_index

            frame_start = frame_end #Update the file pointer
            #The file pointer now points to the start of the next frame
            # --> increment the frame index
            frame_index += 1

            #... then determine if this is the correct frame index or if we need to stride to the next frame
            while (frame_index - START) % STRIDE != 0:
                next_timestep = mm.find(b'ITEM: TIMESTEP', frame_start + 1) #Skip the next adjacent timestep
                if next_timestep == -1:
                    frame_start = file_size # Gracefully hit EOF condition
                    break
                frame_start = next_timestep
                frame_index += 1

        return

    def readFrameLAMMPS(self):

        #Create frame generator
        if self.STEP is None:
            self.FRAME_GENERATOR = self.streamFramesLAMMPS(self.STOP_FRAME,self.START_FRAME,self.FRAME_STRIDE)

        #Advance the frame
        try:
            self.STEP = next(self.FRAME_GENERATOR)
        except StopIteration:
            # Reached the end of the trajectory
            self.PASS = False
            return

    def streamFramesGROMACS(self,STOP,START=0,STRIDE=1):
        #Generator function for fast loading of GROMACS trajectory
        if STOP == -1:
            frame_index = START
            for ts in self.UNIVERSE.trajectory[START::STRIDE]:
                yield frame_index,ts
                frame_index += STRIDE
        else:
            frame_index = START
            for ts in self.UNIVERSE.trajectory[START:STOP:STRIDE]:
                yield frame_index,ts
                frame_index += STRIDE

    def readFrameGROMACS(self):
            
        if self.STEP is None:
            self.FRAME_GENERATOR = self.streamFramesGROMACS(self.STOP_FRAME,self.START_FRAME,self.FRAME_STRIDE)
        
        #Advance the frame
        try:
            self.STEP,ts = next(self.FRAME_GENERATOR)
        except StopIteration:
            # Reached the end of the trajectory
            self.PASS = False
            return
        
        #Unwrap molecules broken by pbc
        #self.UNIVERSE.atoms.unwrap()
        coords = self.UNIVERSE.atoms.positions

        #Update the system's XYZ arrays vectorized
        self.X[:] = coords[:, 0]
        self.Y[:] = coords[:, 1]
        self.Z[:] = coords[:, 2]

        #Update timestep
        self.TIMESTEP = round(ts.time * 1000)
        self.boxLengthX = ts.dimensions[0]
        self.boxLengthY = ts.dimensions[1]
        self.boxLengthZ = ts.dimensions[2]
        
        self.calculateLocalPeakMemUse()

    def createDebugOutFiles(self):

        if self.CONFIG.config['DEBUG']:
            coordOut = None
            self.debugOutFiles = {}
        
            for reactant_index in self.REACTANTS:
                coordOut = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/reactant_{reactant_index}_coordination.txt", 'w')
                self.debugOutFiles[reactant_index] = coordOut

    def completeCutOffList(self):
        #Expands user 'PAIR_CUTOFFS' shortcuts into script-usable cutoffs ((atom_type1,atom_type2)(atom_type3,...):cutoff1) --> (atom_type1,atom_type3):cutoff,(atom_type2,atom_type3):cutoff,...
        #Creates CUTOFF_TARGETS data structure mapping each atom type to any corresponding atom types with an interaction cutoff betweeen them

        new_entries = {}
        for key in self.CONFIG.config['PAIR_CUTOFFS']:
            if isinstance(key[0],tuple) and isinstance(key[1],tuple):
                for item1 in key[0]:
                    for item2 in key[1]:
                        new_entries[(item1,item2)] = self.CONFIG.config['PAIR_CUTOFFS'][key]
                        new_entries[(item2,item1)] = self.CONFIG.config['PAIR_CUTOFFS'][key]
            elif isinstance(key[0],tuple):
                for item in key[0]:
                    new_entries[(item,key[1])] = self.CONFIG.config['PAIR_CUTOFFS'][key]
                    new_entries[(key[1],item)] = self.CONFIG.config['PAIR_CUTOFFS'][key]
            elif isinstance(key[1],tuple):
                for item in key[1]:
                    new_entries[(item,key[0])] = self.CONFIG.config['PAIR_CUTOFFS'][key]
                    new_entries[(key[0],item)] = self.CONFIG.config['PAIR_CUTOFFS'][key]
            else:
                new_entries[(key[0],key[1])] = self.CONFIG.config['PAIR_CUTOFFS'][key]
                new_entries[(key[1],key[0])] = self.CONFIG.config['PAIR_CUTOFFS'][key]

        self.CONFIG.config['PAIR_CUTOFFS'] = new_entries
        
        self.CUTOFF_TARGETS = {}
        for key in self.CONFIG.config['PAIR_CUTOFFS']:
            if key[0] in self.CONFIG.config['ATOM_TYPE_LIST']:
                if key[0] not in self.CUTOFF_TARGETS:
                    self.CUTOFF_TARGETS[key[0]] = []
                if key[1] in self.CONFIG.config['ATOM_TYPE_LIST'] and key[1] not in self.CUTOFF_TARGETS[key[0]]:
                    self.CUTOFF_TARGETS[key[0]].append(key[1])

    def updateReactantExtractClock(self,reactant: molecule,zero=False):
        if zero: #Zero extraction counter to attempt to extract again next iteration
            reactant.extract_wait_steps = 0
            return
        check = not (reactant.extract_wait_steps > 0)
        if check:
            reactant.extract_wait_steps = self.EXTRACTION_TIMESTEP
        else:
            reactant.extract_wait_steps -= self.CONFIG.config['DUMP_FREQ']
        return check

    def clusterGraph(self, reactant: molecule, FIND_COORDINATION:set=None):

        CURRENT_SHELLS = reactant.total_shells

        NEW_SHELL = set()

        for current_molecule_index in FIND_COORDINATION:

            CURRENT_MOL_SHELL = set()

            # Remove the central molecule from the remaining mask immediately
            reactant.remaining_molecule_mask[np.isin(reactant.cluster_mols, current_molecule_index)] = False

            current_molecule = self.MOLECULES[current_molecule_index]

            for current_atom in current_molecule.atoms: #iterate through the atoms in the central atom.

                current_type = self.ATOMTYPES[current_atom]
                
                if current_type not in self.CUTOFF_TARGETS:
                    continue

                for target in self.CUTOFF_TARGETS[current_type]:
                    relevent_atoms = reactant.cluster_atoms[reactant.atom_type_masks[target] & reactant.remaining_molecule_mask]
                    relevent_mols = reactant.cluster_mols[reactant.atom_type_masks[target] & reactant.remaining_molecule_mask]

                    if len(relevent_atoms) == 0:
                        continue

                    #Distance list containing only distances to relevant atoms in cluster (those that are actually in the cutoff list)
                    distance_list = self.allPBCDistance(self.X[current_atom],self.Y[current_atom],self.Z[current_atom],self.X[relevent_atoms],self.Y[relevent_atoms],self.Z[relevent_atoms])

                    cutoff_mask = distance_list <= self.CONFIG.config['PAIR_CUTOFFS'][(current_type,target)]

                    #If any new molecule atoms pass the cutoff test
                    if np.sum(cutoff_mask) > 0:
                        new_mols = relevent_mols[cutoff_mask]
                        new_atoms = relevent_atoms[cutoff_mask]

                        CURRENT_MOL_SHELL.update(new_mols)
                        NEW_SHELL.update(CURRENT_MOL_SHELL-CURRENT_SHELLS)

                        current_molecule.atom_coordinations[current_atom].update(new_mols)

                        for new_atom_index,new_molecule_index in zip(new_atoms,new_mols):
                            self.MOLECULES[new_molecule_index].atom_coordinations[new_atom_index].add(current_molecule_index)

                        #HBond Analysis:

                        distances = distance_list[cutoff_mask]

                        #Identify forward water H-Bonds from the pov of the donator
                        if current_molecule.residue == 'SOL' and self.ELEMENTSYMBOLS[current_atom] == 'H':
                            for atom_index,molecule_index,distance in zip(new_atoms,new_mols,distances):
                                if self.ELEMENTSYMBOLS[atom_index] == 'O' or self.ELEMENTSYMBOLS[atom_index] == 'N' or self.ELEMENTSYMBOLS[atom_index] == 'F':
                                    self.waterHBond(donor_molecule=current_molecule, H_donor_index=current_atom, acceptor_molecule_index=molecule_index, acceptor_atom_index=atom_index, h_bond_radius=distance)

                        #Identify back water H-Bonds from the pov of the acceptor
                        if (self.ELEMENTSYMBOLS[current_atom] == 'O' or self.ELEMENTSYMBOLS[current_atom] == 'N' or self.ELEMENTSYMBOLS[current_atom] == 'F'):
                            for atom_index,molecule_index,distance in zip(new_atoms,new_mols,distances):
                                donating_molecule = self.MOLECULES[molecule_index]
                                if self.ELEMENTSYMBOLS[atom_index] == 'H' and donating_molecule.residue == 'SOL':
                                    self.waterHBond(donor_molecule=donating_molecule, H_donor_index=atom_index, acceptor_molecule_index=current_molecule_index, acceptor_atom_index=current_atom, h_bond_radius=distance)

            #Update non-reactant molecule solvation_shells
            if current_molecule_index != reactant.index:
                current_molecule.updateSolvationShells(0,CURRENT_MOL_SHELL)
                # solvation_shells = current_molecule.solvation_shells
                # if solvation_shells == []:
                #    solvation_shells.append(set())
                # current_molecule.solvation_shells[0].update(CURRENT_MOL_SHELL)

            #Conversely, add index of current central molecule to the first solvation shells of the identified coordinated species
            for molecule_index in CURRENT_MOL_SHELL:
                self.MOLECULES[molecule_index].updateSolvationShells(0,current_molecule_index)
                # molecule = self.MOLECULES[molecule_index]
                # solvation_shells = molecule.solvation_shells
                # if solvation_shells == []:
                #    solvation_shells.append(set())
                # solvation_shells[0].add(current_molecule_index)
            
        self.calculateLocalPeakMemUse()
        return NEW_SHELL

    def reactantGraph(self):

        making_rxn_graph = self.CONFIG.config['CREATE_RXN_GRAPH']

        for reactant_index in self.REACTANTS: #For each cluster...

            reactant = self.MOLECULES[reactant_index]

            if self.CONFIG.configExists('FILTER_REACTANTS_BY_Z'):
                if not self.filterClusterZLocation(reactant_index):
                    reactant.current = None
                    continue

            time_to_extract = False
            extractable_cluster = False
            finished_extracting = self.rankClustersExtracted() #Total extracted cluster number (across ranks) equals 'CLUSTERS_TO_EXTRACT'
            all_atoms_need_coordination = ('QCHEM' in self.CONFIG.config['OUTPUT_TYPE']) if self.CONFIG.configExists('OUTPUT_TYPE') else False #For each molecule added to the cluster, must determine its coordination enviornment as well (for dielectic)

            reactant.buildTotalShells()
            reactant.enumerateCluster(self.MOLECULES) #Expand cluster (composed of moleucle indexes) into 2 parallel lists of atoms and molecules 
            reactant.createTypeMasks(self.ATOMTYPES)
            reactant.createRemainingMolMask()

            if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                self.resetCoordinationConservationDicts()
                self.updateCoordinationConservationTargets(reactant.residue)

            #Start shell building loop:
            CONTINUE = True
            FIND_COORDINATION = {reactant.index}
            EXTRACT_SHELL = None
            LAST_SHELL_LIGANDS = None
            shell_index = 0
            shell_num = 1
            outer_shell_num = shell_num-self.CONFIG.config['REACTION_SHELLS']

            while CONTINUE:
                #Calculate new solvation shell if requested
                if FIND_COORDINATION is not None:
                    SHELL = self.clusterGraph(reactant,FIND_COORDINATION)

                if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                    
                    if (shell_num <= self.CONFIG.config['REACTION_SHELLS']):
                        #Update reactant with new shell
                        reactant.updateSolvationShells(shell_index,SHELL)

                        shell_list = np.array(list(SHELL))
                        shell_res_list = self.RESIDUES[shell_list]
                        ligand_mask = np.isin(shell_res_list,list(self.CONFIG.config['CONSERVE_COORDINATION'].keys()))
                        ligands_res_list = shell_res_list[ligand_mask]
                        for l in ligands_res_list:
                            self.updateCoordinationConservationTargets(l)

                        FIND_COORDINATION = SHELL

                    match outer_shell_num:
                        case 0: #The shell # is now equal to that requested --> determine reactant coordination
                            self.calculateCoordination(reactant)
                            LAST_SHELL_LIGANDS = set(shell_list[ligand_mask]) or None
                            
                            #If the cluster quota has not been reached check for new extractable clusters
                            if not finished_extracting:
                                extractable_cluster = (reactant.current == self.REACTANT_TO_PRINT)
                                if extractable_cluster:
                                    self.REACTANTS_TO_WRITE_FOUND += 1
                                    time_to_extract = self.updateReactantExtractClock(reactant)
                                else:
                                    self.updateReactantExtractClock(reactant,zero=True)
                            else:
                                extractable_cluster = False

                            if extractable_cluster:
                                if time_to_extract:
                                    EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='inner')
                                else:
                                    self.EXTRACT_FAIL_STATS['WRITE_FAILED_OVER_EXTRACT_FREQ'] += 1
                                
                                if EXTRACT_SHELL is None and making_rxn_graph: #If extraction fails at this step but a rxn graph is being made, reduce the coordination find request to only the last shell ligands 
                                    FIND_COORDINATION = LAST_SHELL_LIGANDS
                            elif making_rxn_graph:
                                FIND_COORDINATION = LAST_SHELL_LIGANDS
                            else:
                                FIND_COORDINATION = None
                                CONTINUE = False
                        case 1: #The shell # now contains the full outer shell or only the outer shells of each ligand
                            #Update reactant with new shell
                            reactant.updateSolvationShells(shell_index,SHELL)

                            INNER_SHELLS = reactant.total_shells
                            self.calculateOuterCoordination(reactant,SHELL) #take out
                            if LAST_SHELL_LIGANDS is not None:
                                LIGAND_FIRST_SHELL = {mol for ligand in LAST_SHELL_LIGANDS for mol in self.MOLECULES[ligand].solvation_shells[0]} - INNER_SHELLS
                                REM_OUTER_SHELL = SHELL - LIGAND_FIRST_SHELL

                                if making_rxn_graph:
                                    #self.calculateOuterCoordination(reactant,LIGAND_FIRST_SHELL) #put back
                                    self.countClusterWaterHBonds(reactant,extract_cluster=False,ligand_outer_shells=LIGAND_FIRST_SHELL)

                            if not (extractable_cluster and time_to_extract and not finished_extracting and EXTRACT_SHELL is not None):
                                CONTINUE = False

                            if EXTRACT_SHELL is not None:
                                if LAST_SHELL_LIGANDS is not None:
                                    EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='ligand_first',shell=LIGAND_FIRST_SHELL)
                                else:
                                    EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='outer',shell=REM_OUTER_SHELL)
                                if EXTRACT_SHELL is None:
                                    CONTINUE = False
                                else:
                                    FIND_COORDINATION = EXTRACT_SHELL
                        case 2:
                            #Update reactant with new shell
                            reactant.updateSolvationShells(shell_index,SHELL)
                            
                            if LAST_SHELL_LIGANDS is not None:
                                LIGAND_SECOND_SHELL = {shell_2_mol for shell_1_mol in LIGAND_FIRST_SHELL for shell_2_mol in self.MOLECULES[shell_1_mol].solvation_shells[0]} - INNER_SHELLS - LIGAND_FIRST_SHELL
                                REM_OUTER_SHELL -= LIGAND_SECOND_SHELL
                                EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='ligand_second',shell=LIGAND_SECOND_SHELL)
                                if EXTRACT_SHELL is None:
                                    CONTINUE = False
                                else:
                                    # if all_atoms_need_coordination:
                                    #     FIND_COORDINATION = EXTRACT_SHELL
                                    # else:
                                    #     FIND_COORDINATION = None
                                    FIND_COORDINATION = EXTRACT_SHELL #H-Bond analysis will not work with the above code uncommented, does not find (outer shell) -to- (outer_shell) H-bonds
                            else:
                                CONTINUE = False
                        case 3:
                            #Update reactant with new shell
                            reactant.updateSolvationShells(shell_index,SHELL)

                            EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='outer',shell=REM_OUTER_SHELL)
                            if EXTRACT_SHELL is None:
                                CONTINUE = False
                            else:
                                # if all_atoms_need_coordination:
                                #     FIND_COORDINATION = EXTRACT_SHELL
                                # else:
                                #     FIND_COORDINATION = None
                                FIND_COORDINATION = EXTRACT_SHELL #H-Bond analysis will not work with the above code uncommented, does not find (outer shell) - (outer_shell) H-bonds                        case 4:
                        case 4:    
                            #Update reactant second outer shell
                            reactant.updateSolvationShells(self.CONFIG.config['REACTION_SHELLS']+1,SHELL)

                            CONTINUE = False
                else:
                    #Update reactant with new shell
                    reactant.updateSolvationShells(shell_index,SHELL)

                    match outer_shell_num:
                        case 0: #The shell # is now equal to that requested --> determine reactant coordination
                            self.calculateCoordination(reactant)
                            if not finished_extracting:
                                extractable_cluster = (reactant.coordination == self.CONFIG.config['REACTANT_TO_PRINT']) #self.REACTANT_TO_PRINT)
                                if extractable_cluster:
                                    self.REACTANTS_TO_WRITE_FOUND += 1
                                    time_to_extract = self.updateReactantExtractClock(reactant)
                                else:
                                    self.updateReactantExtractClock(reactant,zero=True)
                            else:
                                extractable_cluster = False

                            if not (extractable_cluster and all_atoms_need_coordination and time_to_extract and not finished_extracting):
                                CONTINUE = False

                            if extractable_cluster:
                                EXTRACT_SHELL = self.createExtractableCluster(reactant,shell_key='inner')
                        case 1: #The shell # now contains the first outer shell
                            CONTINUE = False
                
                    FIND_COORDINATION = SHELL

                reactant.buildTotalShells()
                shell_index += 1
                shell_num += 1
                outer_shell_num += 1

            #Filter clusters that obey H-Bonding statistics
            if reactant.extract_shells is not None:
                #reactant.extract_shells.update(set(mol_list_view))
                self.countClusterWaterHBonds(reactant,extract_cluster=True)
                if self.CONFIG.configExists('HBOND_TARGET'):
                    hbonds_dev = reactant.extract_shells_hbonds - self.CONFIG.config['HBOND_TARGET']
                    if abs(hbonds_dev) > self.CONFIG.config['HBOND_DEV']:
                        if hbonds_dev > 0:
                            self.EXTRACT_FAIL_STATS['WRITE_FAILED_OVER_HBOND_COUNT'] += 1
                        else:
                            self.EXTRACT_FAIL_STATS['WRITE_FAILED_UNDER_HBOND_COUNT'] += 1
                        self.updateReactantExtractClock(reactant,zero=True)
                        reactant.extract_shells = None
                        return None
                if self.CONFIG.configExists('SPECTATOR_TARGET'):
                    spectator_devs = {ion:(reactant.outer_shell[ion] - self.CONFIG.config['SPECTATOR_TARGET'][ion]) for ion in self.CONFIG.config['SPECTATOR_TARGET'].keys()}
                    if any([(abs(spectator_devs[ion]) > self.CONFIG.config['SPECTATOR_DEV'][ion]) for ion in self.CONFIG.config['SPECTATOR_DEV']]):
                        # if hbonds_dev > 0:
                        #     self.EXTRACT_FAIL_STATS['WRITE_FAILED_OVER_HBOND_COUNT'] += 1
                        # else:
                        #     self.EXTRACT_FAIL_STATS['WRITE_FAILED_UNDER_HBOND_COUNT'] += 1
                        self.updateReactantExtractClock(reactant,zero=True)
                        reactant.extract_shells = None
                        return None

        self.calculateLocalPeakMemUse()

    def resetReactantSolvationShells(self):
        #Reset reactant neighborhood solvation shell / atomic coordination data
        for reactant_index in self.REACTANTS:
            reactant = self.MOLECULES[reactant_index]
            for shell in reactant.solvation_shells:
                for molecule_index in shell:
                    if molecule_index not in self.REACTANTS:
                        self.MOLECULES[molecule_index].resetSolvationShells()
            reactant.resetSolvationShells()

    def waterHBond(self, donor_molecule:molecule, H_donor_index:int, acceptor_molecule_index:int, acceptor_atom_index:int, h_bond_radius:float):

        oxygen_index = donor_molecule.atoms[(self.ELEMENTSYMBOLS[donor_molecule.atoms] == 'O')]
        O_donor_index = int(oxygen_index)

        #Using cluster displacement scheme this is easier
        OH_bond_vector = np.array(self.pbcDistance(self.X[H_donor_index],self.X[O_donor_index],self.Y[H_donor_index],self.Y[O_donor_index],self.Z[H_donor_index],self.Z[O_donor_index],returnComponents=True))

        HX_hbond_vector = np.array(self.pbcDistance(self.X[acceptor_atom_index],self.X[H_donor_index],self.Y[acceptor_atom_index],self.Y[H_donor_index],self.Z[acceptor_atom_index],self.Z[H_donor_index],returnComponents=True))

        OH_bond_length = math.sqrt(OH_bond_vector[0]**2 + OH_bond_vector[1]**2 + OH_bond_vector[2]**2)

        #Perform cos(theta) = (u . v) / |u||v| operation to find H-Bond angle
        cos_theta = np.dot(OH_bond_vector, HX_hbond_vector) / (OH_bond_length * h_bond_radius)

        angle = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        if angle <= 45:
            donor_molecule.atom_hbond_donation_atoms[H_donor_index].add((acceptor_molecule_index,acceptor_atom_index))

    def countClusterWaterHBonds(self,reactant:molecule,extract_cluster:bool=True,ligand_outer_shells:set=None):
        hbond_count = 0
        if extract_cluster:
            total_extract_shells = {reactant.index} | reactant.extract_shells
            for molecule_index in total_extract_shells:
                molecule = self.MOLECULES[molecule_index]
                # if not molecule.atom_hbond_donation_atoms:
                #     continue
                hbond_count += sum(acceptor_mol_atom_tuple[0] in total_extract_shells for acceptor_set in molecule.atom_hbond_donation_atoms.values() for acceptor_mol_atom_tuple in acceptor_set)
                #hbond_count += len(molecule.atom_hbond_donation_atoms & (reactant.extract_shells | reactant.index))
        else:
            total_shell = {reactant.index} | {molecule_index for shell in reactant.solvation_shells[:self.CONFIG.config['REACTION_SHELLS']] + [ligand_outer_shells] for molecule_index in shell}
            for molecule_index in total_shell:
                molecule = self.MOLECULES[molecule_index]
                # if not molecule.atom_hbond_donation_atoms:
                #     continue
                hbond_count += sum(acceptor_mol_atom_tuple[0] in total_extract_shells for acceptor_set in molecule.atom_hbond_donation_atoms.values() for acceptor_mol_atom_tuple in acceptor_set)
                #hbond_count += len(molecule.atom_hbond_donation_atoms & total_shell)
            reactant.current.updateHBonds(hbond_count)
        reactant.extract_shells_hbonds = hbond_count

    def filterClusterZLocation(self,reactant_index):
        for bound1,bound2 in self.CONFIG.config['FILTER_REACTANTS_BY_Z']:
            if bound1 < self.POINTPARTICLESZ[reactant_index] < bound2:
                return True
        return False

    def reactantClusterDielectric(self,reactant:molecule):

        def setMoleculeAtomsDielectric(molecule:system.molecule):

            if molecule.atom_coordinations == []:
                raise mySystemError("Attempted to set dielectric for cluster molecule without coordination data")
            
            for atom_index in molecule.atoms:
     
                atom_coordination = molecule.atom_coordinations[atom_index]

                outer_mols = atom_coordination - total_shells

                dielectrics = []

                if outer_mols:
                    for mol in outer_mols:
                        res = self.MOLECULES[mol].residue
                        if res in self.CONFIG.config['RESIDUE_DIELECTRICS']:
                            dielectrics.append(self.CONFIG.config['RESIDUE_DIELECTRICS'][res])
                else:
                    for mol in atom_coordination:
                        res = self.MOLECULES[mol].residue
                        if res in self.CONFIG.config['RESIDUE_DIELECTRICS']:
                            dielectrics.append(self.CONFIG.config['RESIDUE_DIELECTRICS'][res])

                if dielectrics != []:
                    self.ATOM_DIELECTRICS[atom_index] = max(dielectrics)
                else:
                    self.ATOM_DIELECTRICS[atom_index] = 1.0

            molecule_dielectrics = self.ATOM_DIELECTRICS[molecule.atoms]

            if (molecule_dielectrics == 1.0).all():
                raise mySystemError(f"Failed to assign dielectric of any atom in molecule index {molecule_index} in timestep {self.TIMESTEP}, 'RESIDUE_DIELECTRICS' may be incomplete")
            else:
                unassigned_mask = molecule_dielectrics == 1.0
                if sum(unassigned_mask) > 0:
                    molecule_dielectrics[unassigned_mask] = molecule_dielectrics.max()

            #Assign the fancy indexed copy back to the original array
            self.ATOM_DIELECTRICS[molecule.atoms] = molecule_dielectrics

        total_shells = {reactant.index} | reactant.extract_shells

        setMoleculeAtomsDielectric(reactant)

        for molecule_index in reactant.extract_shells:

            molecule = self.MOLECULES[molecule_index]

            setMoleculeAtomsDielectric(molecule)

    def calculateCoordination(self,reactant:molecule):

        graphCoord = []

        #Determine coordination from the residue types of the molecules in the reactants solvation shells
        for shell_index,shell in enumerate(reactant.solvation_shells[:self.CONFIG.config['REACTION_SHELLS']]):

            coordination = {}
            coordination_for_graph = {}

            clusterResidues = self.RESIDUES[list(shell)]

            for residue in self.CONFIG.config['RESIDUE_LIST']:
                res_num = sum(clusterResidues == residue)
                coordination[residue] = res_num
                coordination_for_graph[residue] = res_num

            reactant.coordination[shell_index] = coordination
            graphCoord.append(coordination_for_graph)

        temp = self.GRAPH.rxnNode(configuration=self.CONFIG,solvation_shells=graphCoord)
        if self.CONFIG.config['CREATE_RXN_GRAPH']:
            reactant.current = self.GRAPH.updateGraph(temp, reactant.current)
        else:
            reactant.current = temp

    def calculateOuterCoordination(self,reactant:molecule,outer_shell:set):

        if outer_shell is None:
            raise mySystemError('Unable to calculate outer shell coordination numbers: outer shell empty')

        #Update the current reaction node outer shell averages using outer_shells_total:
        clusterResidues = self.RESIDUES[list(outer_shell)]
        outer_shell_composition = {residue:sum(clusterResidues == residue) for residue in self.CONFIG.config['RESIDUE_LIST']}

        #Temporary
        reactant.outer_shell = {residue:sum(clusterResidues == residue) for residue in self.CONFIG.config['RESIDUE_LIST']}#sum([self.CONFIG.config['RESIDUE_CHARGES'][residue] * count for residue,count in outer_shell_composition.items()])

        reactant.current.updateOuterShell(outer_shell_composition)

    def resetCoordinationConservationDicts(self):
        self.CURRENT_CONSERVED_RESIDUES_TARGETS = {}
        self.CURRENT_CONSERVED_RESIDUES_COUNTS = {}
        for solvent in self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']:
            self.CURRENT_CONSERVED_RESIDUES_TARGETS[solvent] = 0
            self.CURRENT_CONSERVED_RESIDUES_COUNTS[solvent] = 0

    def updateCoordinationConservationTargets(self,specie_key:str):
        #Tally the number of residues requested to be conserved for this particular species
        if self.CONFIG.config['CONSERVE_COORDINATION'][specie_key] is not None:
            for conserved_res in self.CONFIG.config['CONSERVE_COORDINATION'][specie_key]:
                self.CURRENT_CONSERVED_RESIDUES_TARGETS[conserved_res] += self.CONFIG.config['CONSERVE_COORDINATION'][specie_key][conserved_res]
    
    def updateCoordinationConservationCounts(self,specie_key:str,found_count:int):
        #Update the residue counts for the current reactant
        if specie_key not in self.CURRENT_CONSERVED_RESIDUES_COUNTS:
            self.CURRENT_CONSERVED_RESIDUES_COUNTS[specie_key] = found_count
        else:
            self.CURRENT_CONSERVED_RESIDUES_COUNTS[specie_key] += found_count
    
    def checkCoordinationConservation(self):
        if self.CURRENT_CONSERVED_RESIDUES_COUNTS.keys() != self.CURRENT_CONSERVED_RESIDUES_TARGETS.keys():
            self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONSOLVENT'] += 1
            return False
        # zeros = True
        for specie,count in self.CURRENT_CONSERVED_RESIDUES_TARGETS.items():
            if self.CURRENT_CONSERVED_RESIDUES_COUNTS[specie] > count:
                return False
            # if self.CURRENT_CONSERVED_RESIDUES_COUNTS[specie] < count:
            #     zeros = False
        # if zeros:
        #     return 0
        return True

    def peekCoordinationConservation(self,specie_key:str,found_count:int):
        return (self.CURRENT_CONSERVED_RESIDUES_COUNTS[specie_key] + found_count) - self.CURRENT_CONSERVED_RESIDUES_TARGETS[specie_key]

    def createExtractableCluster(self,reactant:molecule,shell_key:str='',shell:set=None):

        #Uses self.CONFIG.configExists('CONSERVE_COORDINATION') (if it exists) to expand the coordination shells of the current reactant such that important ligands have 1st solvation shells with the aim of conserving a certain composition.
        #Returns the expanded portion of the cluster (such that the first solvation shell(s) of this expanded portion may be found for QCHEM dielectric assignment)
        
        if shell_key not in ('inner','ligand_first','ligand_second','outer'):
            self.CONFIG.STDOUT.write('WARNING: Calling createExtractableCluster() with invalid shell_key\n')
            return

        if reactant.extract_shells is None:
            reactant.extract_shells = set()

        if shell is None:
            NEW_SHELL = reactant.solvation_shells
        else:
            NEW_SHELL = shell
        
        if self.CONFIG.configExists('CONSERVE_COORDINATION'):                                                                                                  

            if shell_key == 'inner':
                #For each residue whose composition must be conserved check for conservation
                for shell in NEW_SHELL:
                    for conserved_res in self.CURRENT_CONSERVED_RESIDUES_TARGETS:
                        #Calculate the difference between the conserved residue number and the actual
                        res_counts = sum(self.RESIDUES[list(shell)] == conserved_res)
                        self.updateCoordinationConservationCounts(conserved_res,res_counts)
                    if not self.checkCoordinationConservation():
                        self.updateReactantExtractClock(reactant,zero=True)
                        self.calculateLocalPeakMemUse()
                        reactant.extract_shells = None
                        self.EXTRACT_FAIL_STATS['WRITE_FAILED_INNER_NONCONSERVE'] += 1
                        return None
            else:
                shell_list = np.array(list(NEW_SHELL))
                shell_residues = self.RESIDUES[shell_list]
                outer_shell_by_res = {res:shell_list[shell_residues == res] for res in set(shell_residues)}

                if shell_key == 'ligand_first':
                    #First outer solvation shell contains non-conserved residues making it imposible to complete, exit
                    for residue_type,molecule_index_list in outer_shell_by_res.items():
                        self.updateCoordinationConservationCounts(residue_type,len(molecule_index_list))
                    current_non_solvent_failed_count = self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONSOLVENT']
                    if not self.checkCoordinationConservation():
                        if self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONSOLVENT'] == current_non_solvent_failed_count:
                            self.EXTRACT_FAIL_STATS['WRITE_FAILED_FIRSTOUTER_NONCONSERVE'] += 1
                        self.updateReactantExtractClock(reactant,zero=True)
                        self.calculateLocalPeakMemUse()
                        reactant.extract_shells = None
                        return None
                else:
                    for residue_type,molecule_index_list in outer_shell_by_res.items():
                        if residue_type not in self.CURRENT_CONSERVED_RESIDUES_TARGETS or self.peekCoordinationConservation(residue_type,0) == 0: #Residue type is not one of the conserved types ...or... the residue count already conserves composition --> prune and continue
                            NEW_SHELL = NEW_SHELL - set(molecule_index_list) 
                            continue

                        res_count = len(molecule_index_list)
                        overage = self.peekCoordinationConservation(residue_type,res_count)
                        
                        if overage < 0: #Residue number is below the number required to maintain composition, and no more shells remain to add, exit
                            if shell_key == 'outer':
                                self.updateReactantExtractClock(reactant,zero=True)
                                self.calculateLocalPeakMemUse()
                                reactant.extract_shells = None
                                self.EXTRACT_FAIL_STATS['WRITE_FAILED_SECONDOUTER_UNDERCOORD'] += 1
                                return None
                            if shell_key == 'ligand_second':
                                self.updateCoordinationConservationCounts(residue_type,len(molecule_index_list))
                                continue
                        elif overage == 0:
                            self.updateCoordinationConservationCounts(residue_type,len(molecule_index_list))
                            continue
                        else:
                            #Residue number exceeds number required to maintain composition, start pruning this residue from NEW_SHELL
                            popped_set = set()

                            mol_list_view = molecule_index_list[::]

                            #Shell pruning loop, prunes by residue to conserve composition
                            while overage > 0 and len(mol_list_view) > 0:

                                popped_set.add(mol_list_view[0])
                                mol_list_view = mol_list_view[1:]
                                
                                overage -= 1

                            self.updateCoordinationConservationCounts(residue_type,len(mol_list_view))
                            NEW_SHELL = NEW_SHELL - popped_set
        else:
            for shell in NEW_SHELL:
                if any(np.isin(self.ATOMTYPES[list(shell)],self.CONFIG.config['EXCLUDE_SOLVENT_RESIDUES'])) > 0:
                    self.updateReactantExtractClock(reactant,zero=True)
                    self.calculateLocalPeakMemUse()
                    reactant.extract_shells = None
                    self.EXTRACT_FAIL_STATS['WRITE_FAILED_INNER_EXCLUDE_RES'] += 1
                    return None

        if isinstance(NEW_SHELL,list):
            for shell in NEW_SHELL[:self.CONFIG.config['REACTION_SHELLS']]:
                reactant.extract_shells.update(shell)
        elif isinstance(NEW_SHELL,set):
            reactant.extract_shells.update(NEW_SHELL)
        self.calculateLocalPeakMemUse()
        return NEW_SHELL

    def writeCoordination(self,reactant,outfile=None):
        
        reactant_atoms = ' '.join([str(atom) for atom in reactant.atoms])

        if outfile is None:
            outfile = self.debugOutFiles[reactant.index]
        
        write = ''

        #write = f"[TIMESTEP]\n{self.TIMESTEP}\n\n"

        #write += f"[COORDINATION NUMBERS]\n"

        # for shell_index,shell in enumerate(reactant.coordination):
        #     write += f"[Shell #]\n{shell_index+1}\n[RES] [COORDINATION NUMBER]\n"
        #     for res,coordination in shell.items():
        #         write += f"{res} {coordination}\n"
        #     write += '\n'

        #rremove
        write += f"{reactant.coordination[0]['SOL']} {reactant.coordination[1]['SOL']}\n"

        # write += f"[SOLVATION SHELLS]\n[Shell 0] {reactant_atoms}\n"

        # for shell_index, shell in enumerate(reactant.solvation_shells[:self.CONFIG.config['REACTION_SHELLS']]):
        #     write += f"[Shell {shell_index+1}] "
        #     for molecule_index in shell:
        #         molecule = self.MOLECULES[molecule_index]
        #         write += ' '.join([str(atom) for atom in molecule.atoms]) + ' '
        #     write += '\n'
        # write += '\n'

        # write += f"[CLUSTER]\n"

        # for molecule_index in reactant.cluster:
        #     molecule = self.MOLECULES[molecule_index]
        #     for atom_index in molecule.atoms:
        #         write += f"{atom_index} "

        #write += '\n\n'

        outfile.write(write)

    def writeRxnNetwork(self):

        self.graphAggregate()

        if self.NP > 1 and self.RANK != 0:
            return
        
        #Overwrite graphOut to only show most recent reaction graph
        self.graphOut = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/rxn_graph.{self.STEP}.txt",'w')
        self.ALL_RANKS_GRAPH.writeWeights(self.graphOut, start=self.START_STEP, end=self.STEP)
        self.graphOut.flush()

        self.calculateLocalPeakMemUse()

    def allPBCDistance(self,xi,yi,zi,xj_array,yj_array,zj_array, returnComponents=False):
        #Find radial distance between point i (defined by xyz) and points j (defined by xj_array,yj_array, and zj_array) and returns distance array equal in length to j arrays using pbc distance finding snippet from Hemanth Haridas
        #dA = absolute value (Ai - Aj) - L * round( (Ai - Aj) / L))
        #...where A is the coordinate X,Y, or Z and L is the box length

        try:
            assert isinstance(xj_array,np.ndarray); isinstance(yj_array,np.ndarray); isinstance(zj_array,np.ndarray), "ERROR in allPBCDistance, recieved non ndarray"  
        except AssertionError:
            raise mySystemError('allPBCDistance called using non numpy ndarray object')
        
        try:
            assert xj_array.size == yj_array.size == zj_array.size, "ERROR in allPBCDistance, recieved arrays of unequal length"
        except AssertionError:
            return

        # Calculate the raw differences by subtracting the single point from each array
        dX = xi - xj_array #dX is new numpy array broadcasted from xi - xj_array
        dY = yi - yj_array
        dZ = zi - zj_array
        
        # Use NumPy vectorized rounding and periodic boundary conditions
        dXpbc = dX - self.boxLengthX * np.rint(dX / self.boxLengthX)
        dYpbc = dY - self.boxLengthY * np.rint(dY / self.boxLengthY)
        dZpbc = dZ - self.boxLengthZ * np.rint(dZ / self.boxLengthZ)

        self.calculateLocalPeakMemUse()

        if returnComponents:
            return dXpbc, dYpbc, dZpbc
        else:
            # Use NumPy's vectorized square root for maximum performance
            return np.sqrt(dXpbc**2 + dYpbc**2 + dZpbc**2)    

    def pbcDistance(self,xi,xj,yi,yj,zi,zj,returnComponents=False):
        #Find radial distance between point i and point j using pbc distance finding snippet from Hemanth Haridas
        #dA = absolute value (Ai - Aj) - L * round( (Ai - Aj) / L))
        #...where A is the coordinate X,Y, or Z and L is the box length

        #Note: When returnComponents=False order for i and j does NOT matter, pbcDistance will return the same absolute PBC distance regardless of which species is chosen to be i or j

        #Note: When returnComponents=True order for i and j DOES matter, the distance returned will be the dX, dY, and dZ of particle i wrt particle j

        dX = xi - xj
        dXpbc = dX - self.boxLengthX * round(dX / self.boxLengthX)

        dY = yi - yj
        dYpbc = dY - self.boxLengthY * round(dY / self.boxLengthY)

        dZ = zi - zj
        dZpbc = dZ - self.boxLengthZ * round(dZ / self.boxLengthZ)

        if returnComponents:
            return dXpbc, dYpbc, dZpbc
        else:
            return math.sqrt(dXpbc**2 + dYpbc**2 + dZpbc**2)

    def outputCheck(self):

        #fix
        if self.RANK == 0 and self.STEP >= 50:
    
            self.graphOut = open('./' + self.CONFIG.config['WRITE_DIRECTORY'] + '/' + self.CONFIG.config['RUN_NAME'] + '/' +'rank_0_graph.txt','r')

            compare = open('./graph_save_Li.txt')

            unexpected = 0

            line1 = self.graphOut.readlines()
            line2 = compare.readlines()

            if sorted(line1) != sorted(line2):
                unexpected = 1

            # while line1 and line2:
                
            #     if line1 != line2:
            #         unexpected = 1

            #     line1 = self.graphOut.readline()
            #     line2 = compare.readline()

            if unexpected:
                if self.RANK == 0:
                    self.CONFIG.STDOUT.write("UNEXPECTED OUTPUT ENCOUNTERED\n")
            else:
                if self.RANK == 0:
                    self.CONFIG.STDOUT.write("NO PROBLEMS ENCOUNTERED\n")

    def initLAMMPS(self):

        self.parseDataLAMMPS()
        self.connectivityLAMMPS()
        self.line = None
        self.checkConfiguration('RESIDUE_LIST')
        self.assignResidues()
        self.wrapMoleculesWrapperLAMMPS()

    def initGROMACS(self):

        self.parseDataAndConnectivityGROMACS()
        residue_list = self.createResidueList()
        self.checkConfiguration('RESIDUE_LIST',residue_list)
        self.loadXTCGuarded()

    def parseBondsFromTOP(self):
        """
        Parses a GROMACS .top file, extracts molecule-specific bond and atom
        mass information, and adds them to the MDAnalysis Universe based on
        the system's atom mapping.

        """

        if not self.CONFIG.configExists('TOPOLOGY_FILE_PATH'):
            self.CONFIG.STDOUT.write('WARNING: Bond parsing attempted without TOPOLOGY_FILE_PATH\n')
            return

        # 1. Parse the entire .top file to get molecule definitions
        molecule_defs = {}
        current_mol = None

        lines = self.TOPO_FILE.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('[ moleculetype ]'):
                i += 2
                mol_line = lines[i].strip()
                mol_name = mol_line.split()[0]
                molecule_defs[mol_name] = {'bonds': [], 'atom_indices': [], 'atom_types':[], 'masses': []}
                current_mol = mol_name
            elif current_mol and line.startswith('[ atoms ]'):
                i += 2
                while i < len(lines) and not lines[i].strip().startswith('['):
                    parts = lines[i].strip().split()
                    if len(parts) > 1 and not parts[0].startswith(';'):
                        gromacs_idx = int(parts[0])
                        mass = float(parts[-1])
                        Type = str(parts[4])
                        molecule_defs[current_mol]['atom_indices'].append(gromacs_idx)
                        molecule_defs[current_mol]['atom_types'].append(Type)
                        molecule_defs[current_mol]['masses'].append(mass)
                    i += 1
            elif current_mol and line.startswith('[ bonds ]'):
                i += 2
                while i < len(lines) and not lines[i].strip().startswith('['):
                    parts = lines[i].strip().split()
                    if len(parts) >= 2 and not parts[0].startswith(';'):
                        atom1 = int(parts[0])
                        atom2 = int(parts[1])
                        molecule_defs[current_mol]['bonds'].append((atom1, atom2))
                    i += 1
            else:
                i += 1
        
        # 2. Iterate through the system's molecules, map bonds and masses
        system_bonds = []
        
        # Create a new, empty masses array for the Universe
        new_masses = np.zeros(self.UNIVERSE.atoms.n_atoms)

        for mol in self.MOLECULES:
            mol_name = mol.residue
            if mol_name in molecule_defs:
    
                mol_def = molecule_defs[mol_name]

                assert (mol_def['atom_types'] == self.ATOMTYPES[mol.atoms]).all(),'ERROR: Mismatched residue names in .top and .gro files'
                
                for bond in mol_def['bonds']:
                    system_bonds.append((mol.atoms[bond[0]-1], mol.atoms[bond[1]-1]))

                if len(mol_def['masses']) != len(mol.atoms):
                    self.CONFIG.STDOUT.write('Warning: .top mass format does not match .gro format\n')
                else:
                    for mass in range(len(mol_def['masses'])):
                        new_masses[mol.atoms[mass]] = mol_def['masses'][mass]
            else:
                self.CONFIG.STDOUT.write('Warning: Unrecognized .top molecule definition encountered\n')

        # 3. Add bonds and masses to the MDAnalysis Universe
        if not system_bonds:
            if self.RANK == 0:
                self.CONFIG.STDOUT.write('Warning: No bonds found or added.. check .top or .gro\n')
        else:
            self.UNIVERSE.add_bonds(system_bonds)
            if self.RANK == 0:
                self.CONFIG.STDOUT.write(f"Successfully added {len(system_bonds)} bonds to system from {self.CONFIG.config['TOPOLOGY_FILE_PATH']}\n")
        
        if not np.all(new_masses):
            if self.RANK ==0:
                self.CONFIG.STDOUT.write('Warning: Some atoms have zero mass\n')
        else:
            self.UNIVERSE.atoms.masses = new_masses
            if self.RANK == 0:
                self.CONFIG.STDOUT.write(f"Successfully added masses to system from {self.CONFIG.config['TOPOLOGY_FILE_PATH']}\n")

    def loadXTCGuarded(self):

        #Stolen from Jackson Elowitt
        self.UNIVERSE = None

        #Rank 0 creates universe first to generate index file and prevent file writing race conditions
        if self.RANK == 0:
            
            self.CONFIG.STDOUT.write(f"Loading {self.CONFIG.config['TRJ_FILE_PATH']}\n")
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.topology.guessers')
            self.UNIVERSE = mda.Universe(self.CONFIG.config['DATA_FILE_PATH'],self.CONFIG.config['TRJ_FILE_PATH'])

        #Synchronize all processes
        self.COMM.Barrier()

        if self.RANK != 0:
            self.UNIVERSE = mda.Universe(self.CONFIG.config['DATA_FILE_PATH'],self.CONFIG.config['TRJ_FILE_PATH'])

        if isinstance(self.CONFIG.config['FRAMES_TO_PROCESS'],tuple):
            self.UNIVERSE.trajectory

        self.COMM.Barrier()

        self.calculateLocalPeakMemUse()

    def calculateProcessingRate(self):
        if self.RANK == 0:
            
            check_time = time.time()

            run_time = check_time - self.START_TIME

            self.PROCESSING_RATE = round((((self.STEP+1)*self.CONFIG.config['DUMP_FREQ']/1000000)/(run_time/86400)),3)

    def calculateLocalPeakMemUse(self,reset=False):
        if reset:
            self.CURRENT_LOCAL_MEMORY_USE = 0
        current_mem = np.int64(self.PROCESS.memory_info().rss) #Get current process memory use as a 64-bit integer
        self.CURRENT_LOCAL_MEMORY_USE = max(current_mem,self.CURRENT_LOCAL_MEMORY_USE)

    def calculateAllProcPeakMemUse(self):
        #Creates send buffer on all proc. and recieve buffer on root proc. for the current process memory use, distributes each memory uses to root, sums each process current memory use to get the total current memory use, and updates the current all-process peak memory use if needed
        all_current = np.array(0, dtype=np.int64) #Create all-process memory use space on all processes (must initiallize on all processes to prevent 'None' errors)
        self.COMM.Reduce([self.CURRENT_LOCAL_MEMORY_USE,MPI.INT64_T],[all_current,MPI.INT64_T],op=MPI.SUM, root=0) #Synchronous send-recieve call and summation of all memory uses to root proc.
        if self.RANK == 0:
            all_current = all_current/1e9 #Sum current all-process memory use and convert to GB
            self.PEAK_MEMORY_USE = round(max(all_current, self.PEAK_MEMORY_USE),3) #Update peak memory use if needed

    def printProgress(self):
        if self.RANK == 0:
            if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                #self.CONFIG.STDOUT.write(f'FRAMES ANALYZED: {self.STEP}    CLUSTERS EXTRACTED: {self.TOTAL_CLUSTERS_WRITTEN}/{self.TOTAL_CLUSTERS_TO_EXTRACT}, {self.TOTAL_REACTANTS_TO_WRITE_FOUND-self.TOTAL_CLUSTERS_WRITTEN} failed: {self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_OVER_EXTRACT_FREQ']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_INNER_NONCONSERVE']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_FIRSTOUTER_NONCONSERVE']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_FIRSTOUTER_NONSOLVENT']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_SECONDOUTER_UNDERCOORD']}    PROCESSING RATE: {self.PROCESSING_RATE} ns/day    PEAK MEMORY USE: {self.PEAK_MEMORY_USE} GB\n')
                self.CONFIG.STDOUT.write(f"FRAME: {self.STEP} (0 indexed)    EXTRACTED: {self.TOTAL_CLUSTERS_WRITTEN}/{self.TOTAL_CLUSTERS_TO_EXTRACT}, {self.TOTAL_REACTANTS_TO_WRITE_FOUND-self.TOTAL_CLUSTERS_WRITTEN} failed: {self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_INNER_NONCONSERVE']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_FIRSTOUTER_NONCONSERVE']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_SECONDOUTER_UNDERCOORD']}    RATE: {self.PROCESSING_RATE} ns/day    PEAK MEM: {self.PEAK_MEMORY_USE} GB\n")
            else:
                self.CONFIG.STDOUT.write(f"FRAMES ANALYZED: {self.STEP} (0 indexed)    CLUSTERS EXTRACTED: {self.TOTAL_CLUSTERS_WRITTEN}/{self.TOTAL_CLUSTERS_TO_EXTRACT}, {self.TOTAL_REACTANTS_TO_WRITE_FOUND-self.TOTAL_CLUSTERS_WRITTEN} failed: {self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_OVER_EXTRACT_FREQ']}/{self.EXTRACT_FAIL_STATS['TOTAL_WRITE_FAILED_INNER_EXCLUDE_RES']}    PROCESSING RATE: {self.PROCESSING_RATE} ns/day    PEAK MEMORY USE: {self.PEAK_MEMORY_USE} GB\n")

    def run(self):

        if self.CONFIG.config['SYSTEM_TYPE'] == 'lammps':
            self.readFrameLAMMPS()
            self.wrapMoleculesWrapperLAMMPS()
        elif self.CONFIG.config['SYSTEM_TYPE'] == 'gromacs':
            self.readFrameGROMACS()

        while self.PASS:

            if self.STEP > 0:
                self.calculateLocalPeakMemUse(reset=True)

            self.reactantClustersWrapper()
            self.reactantGraph()
            if self.CONFIG.config['CREATE_RXN_GRAPH'] and (self.STEP * self.CONFIG.config['DUMP_FREQ'] % self.GRAPH_WRITE_TIMESTEP) == 0 and self.STEP != self.START_FRAME:
                self.writeRxnNetwork()
            self.extractReactantClusters()
            self.gatherClusterExtractionStats()
            self.calculateProcessingRate()
            self.calculateAllProcPeakMemUse()
            self.resetReactantSolvationShells()

            self.printProgress()

            if not self.CONFIG.config['CREATE_RXN_GRAPH'] and self.allRankClustersExtracted():
                self.PASS = False
            else:
                if self.CONFIG.config['SYSTEM_TYPE'] == 'lammps':
                    self.readFrameLAMMPS()
                    self.wrapMoleculesWrapperLAMMPS()
                elif self.CONFIG.config['SYSTEM_TYPE'] == 'gromacs':
                    self.readFrameGROMACS()

    def initialize(self):

        #Check initial configurations (these must be done one at a time because there is a subset of configuratons that must be checked at a later point)
        first_check_params = ['SYSTEM_TYPE','DATA_FILE_PATH','TRJ_FILE_PATH','DUMP_FREQ','FRAMES_TO_PROCESS','REACTION_SHELLS','CLUSTER_MOLECULES','OUTPUT_TYPE','CLUSTERS_TO_EXTRACT']
        for param in first_check_params:
            self.checkConfiguration(param)

        if self.CONFIG.config['SYSTEM_TYPE'] == 'lammps':

            self.initLAMMPS()

        elif self.CONFIG.config['SYSTEM_TYPE'] == 'gromacs':
        
            self.initGROMACS()

        second_check_params = ['DUMP_FREQ','FRAMES_TO_PROCESS','REACTANT','REACTION_SHELLS','CLUSTER_MOLECULES','PAIR_CUTOFFS','CREATE_RXN_GRAPH','WRITE_DIRECTORY','RUN_NAME','OUTPUT_TYPE','CLUSTERS_TO_EXTRACT','REACTANT_TO_PRINT','CONSERVE_COORDINATION','COARSEN_SOLVENT_RESIDUES','EXCLUDE_SOLVENT_RESIDUES','FILTER_REACTANTS_BY_Z','HBOND_TARGET','HBOND_DEV','SPECTATOR_TARGET','SPECTATOR_DEV','RESIDUE_DIELECTRICS','RESIDUE_CHARGES','QC_BASIS','QC_BASIS2','QC_PSEUDO','QC_METHOD','QC_PCM_METHOD','PROFILE','DEBUG']
        for param in second_check_params:
            self.checkConfiguration(param)

        self.makeElementSymbolList()
        self.createReactantList()
        self.createDebugOutFiles()

        self.POINTPARTICLESX = np.zeros(self.NUM_MOLECULES,dtype=float)
        self.POINTPARTICLESY = np.zeros(self.NUM_MOLECULES,dtype=float)
        self.POINTPARTICLESZ = np.zeros(self.NUM_MOLECULES,dtype=float)
        self.ALL_POINTPARTICLESXYZ = np.zeros(self.NUM_MOLECULES*3,dtype=float)

        self.RESIDUES = [self.MOLECULES[molecule_index].residue for molecule_index in range(len(self.MOLECULES))]
        self.RESIDUES = np.array(self.RESIDUES)

    def finallize(self):

        if self.CONFIG.config['CREATE_RXN_GRAPH']:
            self.writeRxnNetwork()
        
        #self.outputCheck() #temporary
        
        tracemalloc.stop()

        MPI.Finalize()

    def writeBoxToXYZ(self):

        if self.RANK == 0:

            xyz = open('./' + self.CONFIG.config['WRITE_DIRECTORY'] + '/' + self.CONFIG.config['RUN_NAME'] + '/box.xyz','w')

            xyz.write(str(self.NUM_ATOMS) + '\n' + 'Current XYZ for project: ' + self.CONFIG.config['REACTANT'] + '\n')

            for mol in self.MOLECULES:

                for atom in mol.atoms:

                    xyz.write(str(self.ELEMENTSYMBOLS[atom]) + " " + str(self.X[atom]) + " " + str(self.Y[atom]) + " " + str(self.Z[atom]) + '\n')

            xyz.flush()

    def writeSolvationShellsToXYZ(self,reactant:molecule):

        xyz = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/solvation_shell_{reactant.index}_{self.TIMESTEP}.xyz",'w')

        atom_num = 0
        write = ''

        for j in reactant.atoms:
            atom_num += 1

            write += (str(self.ELEMENTSYMBOLS[j][:1]) + " " + str(self.X[j]) + " " + str(self.Y[j]) + " " + str(self.Z[j]) + '\n')

        refX = self.POINTPARTICLESX[reactant.index]
        refY = self.POINTPARTICLESY[reactant.index]
        refZ = self.POINTPARTICLESZ[reactant.index]

        for shell in reactant.solvation_shells:
            for molecule_index in shell:
                for atom in self.MOLECULES[molecule_index].atoms:
                    atom_num += 1

                    relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom],refX,self.Y[atom],refY,self.Z[atom],refZ,returnComponents=True)

                    write += (str(self.ELEMENTSYMBOLS[atom][:1]) + " " + str(refX+relativeX) + " " + str(refY+relativeY) + " " + str(refZ+relativeZ) + '\n')

        xyz.write(f"{atom_num}\nSolvation Shells of Reactant Index: {reactant.index} at Timestep {self.TIMESTEP}\n{write}")

        xyz.flush()

    def writeClusterToXYZ(self,reactant:molecule):

        xyz = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/cluster_{reactant.index}_{self.TIMESTEP}.xyz",'w')

        atom_num = 0
        write = ''

        for j in reactant.atoms:
            atom_num += 1
            write += (str(self.ELEMENTSYMBOLS[j][:1]) + " " + str(self.X[j]) + " " + str(self.Y[j]) + " " + str(self.Z[j]) + '\n')

        refX = self.POINTPARTICLESX[reactant.index]
        refY = self.POINTPARTICLESY[reactant.index]
        refZ = self.POINTPARTICLESZ[reactant.index]

        for molecule_index in reactant.cluster:
            for atom in self.MOLECULES[molecule_index].atoms:
                atom_num += 1

                relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom],refX,self.Y[atom],refY,self.Z[atom],refZ,returnComponents=True)
                write += (str(self.ELEMENTSYMBOLS[j][:1]) + " " + str(self.X[j]) + " " + str(self.Y[j]) + " " + str(self.Z[j]) + '\n')

        xyz.write(f"{atom_num}\nSolvation Shells of Reactant Index: {reactant.index} at Timestep {self.TIMESTEP}\n{write}")

        xyz.flush()

    def writeAllClustersToXYZ(self):

        xyz = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/all_clusters_{self.TIMESTEP}.xyz",'w')

        atom_num = 0
        write = ''

        for reactant_index in self.REACTANTS:

            reactant = self.MOLECULES[reactant_index]

            for j in reactant.atoms:
                atom_num += 1

                write += (str(self.ELEMENTSYMBOLS[j][:1]) + " " + str(self.X[j]) + " " + str(self.Y[j]) + " " + str(self.Z[j]) + '\n')

            refX = self.POINTPARTICLESX[reactant.index]
            refY = self.POINTPARTICLESY[reactant.index]
            refZ = self.POINTPARTICLESZ[reactant.index]

            for molecule_index in reactant.cluster:
                for atom in self.MOLECULES[molecule_index].atoms:
                    atom_num += 1

                    relativeX, relativeY, relativeZ = self.pbcDistance(self.X[atom],refX,self.Y[atom],refY,self.Z[atom],refZ,returnComponents=True)

                    write += (str(self.ELEMENTSYMBOLS[j][:1]) + " " + str(self.X[j]) + " " + str(self.Y[j]) + " " + str(self.Z[j]) + '\n')

        xyz.write(f"{atom_num}\nSolvation Shells of Reactant Index: {reactant.index} at Timestep {self.TIMESTEP}\n{write}")

        xyz.flush()

    def graphAggregateIO(self):

        RESIDUE_REGEX = re.compile(r'([A-Z0-9a-z]+):(\d+)') #Parses RES:X shell residue
        SOLVENT_REGEX = re.compile(r'([A-Z0-9a-z]+)\((\d+)-(\d+)\)') #Parses RES(X-Y) shell residue range

        def parseEdgeLine(line):

            if not line:
                return None,None

            bidirectional = False

            edge_split_string1 = '-->'
            edge_split_string2 = '<-->'
            shell_split_string = '\[Shell .*?\]' #Regex for capturing [Shell XXX] pattern using non-greedy, any-length wildcard match (.*?)

            line = line.strip()

            if edge_split_string2 in line:
                node_string1,node_string2 = line.split(edge_split_string2,1)
                bidirectional = True
            else:
                node_string1,node_string2 = line.split(edge_split_string1,1)

            node_string1 = node_string1.strip()
            node_string2 = node_string2.strip()

            node_shells1 = [item.strip() for item in re.split(shell_split_string,node_string1) if item]
            node_shells2 = [item.strip() for item in re.split(shell_split_string,node_string2) if item]

            node_residues1 = [shell.split() for shell in node_shells1]
            node_residues2 = [shell.split() for shell in node_shells2]

            node_contents1 = []
            node_contents2 = []
            solvent_range1 = []
            solvent_range2 = []
    
            for shell1,shell2 in zip(node_residues1,node_residues2):
                residue_dict = {}
                solvent_ranges = {}
                for residue in shell1:
                    match = RESIDUE_REGEX.match(residue)
                    if match:
                        residue_dict[match.group(1)] = int(match.group(2))
                    match = SOLVENT_REGEX.match(residue)
                    if match:
                        solvent_ranges[match.group(1)] = {int(match.group(2)),int(match.group(3))}
                node_contents1.append(residue_dict)
                solvent_range1.append(solvent_ranges)
                
                residue_dict = {}
                solvent_ranges = {}
                for residue in shell2:
                    match = RESIDUE_REGEX.match(residue)
                    if match:
                        residue_dict[match.group(1)] = int(match.group(2))
                    match = SOLVENT_REGEX.match(residue)
                    if match:
                        solvent_ranges[match.group(1)] = {int(match.group(2)),int(match.group(3))}
                node_contents2.append(residue_dict)
                solvent_range2.append(solvent_ranges)

            node1 = rxnGraph.rxnNode(configuration=self.CONFIG,solvation_shells=node_contents1,sol_counts=solvent_range1)
            node2 = rxnGraph.rxnNode(configuration=self.CONFIG,solvation_shells=node_contents2,sol_counts=solvent_range2)

            return node1,node2,bidirectional
        
        TEMP = rxnGraph(self.CONFIG)

        for rank in range(0,self.NP):
            fileName = ''.join(['./',str(self.CONFIG.config['WRITE_DIRECTORY']),'/',str(self.CONFIG.config['RUN_NAME']),'/rank_',str(rank),'_graph.txt'])

            graphIn = open(fileName,'r')
            
            lines = graphIn.readlines()

            for line in lines[1:]:

                reactant,product,bidirectional = parseEdgeLine(line)

                if not reactant or not product:
                    raise mySystemError(f'graphAggregateIO failed to parse graph line in {fileName}')
                
                if (reactant == product):
                    raise mySystemError(f'graphAggregateIO encoutered self-referencing graph edge')

                TEMP.addEdge(reactant,product,bidirectional)

        self.graphOut = open('./' + self.CONFIG.config['WRITE_DIRECTORY'] + '/' + self.CONFIG.config['RUN_NAME'] + '/full_graph.txt','w')
        TEMP.writeRxnGraph(self.graphOut, self.TIMESTEP)
        self.graphOut.flush()

    def graphAggregate(self):

        if self.NP == 1:
            self.ALL_RANKS_GRAPH = self.GRAPH
            return
        
        graph_len = self.GRAPH.getSize()

        graph_data = self.GRAPH.serialize(self.RANK)
        graph_data_size = graph_data.size

        graph_size_metrics = np.array([graph_len,graph_data_size],dtype=np.int32)

        all_graph_size_metrics = np.empty(self.NP*graph_size_metrics.size,dtype=np.int32)
        self.COMM.Gather([graph_size_metrics,MPI.INT],[all_graph_size_metrics,MPI.INT])

        ALL_RANKS_DATA = self.arrangeBuffer(graph_data,sendDataType=graph_data.dtype,ALL=False)

        if self.RANK == 0:
            #Clear all-rank graph object
            self.ALL_RANKS_GRAPH = rxnGraph(self.CONFIG)

            all_res_keys_sorted = sorted(set(self.CONFIG.config['RESIDUE_LIST'].keys()))
            res_keys_sorted = sorted(set(self.CONFIG.config['RESIDUE_LIST'].keys()) - (set() if not self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES') else set(self.CONFIG.config['COARSEN_SOLVENT_RESIDUES'])))
            solvent_keys_sorted = sorted((set() if not self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES') else set(self.CONFIG.config['COARSEN_SOLVENT_RESIDUES'])))
            
            all_res_len = len(all_res_keys_sorted)
            res_len = len(res_keys_sorted)
            sol_len = len(solvent_keys_sorted)
            num_shells = self.CONFIG.config['REACTION_SHELLS']
            shell_len = res_len + sol_len
            if all_res_len != shell_len:
                raise mySystemError('Graph unpacking chunk size issue, check residue list')
            current = 0

            metrics = all_graph_size_metrics.reshape(-1, 2)

            for rank_graph_len, rank_data_size in metrics:
                
                if rank_graph_len == 0 or rank_data_size == 0:
                    current += rank_data_size
                    continue

                rank_data = ALL_RANKS_DATA[current : current + rank_data_size]
                
                node_data_end = rank_graph_len * num_shells * shell_len
                if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
                    node_data_end += rank_graph_len * shell_len
                if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
                    node_block = rank_data[:node_data_end].reshape(rank_graph_len, num_shells+1, shell_len)
                else:
                    node_block = rank_data[:node_data_end].reshape(rank_graph_len, num_shells, shell_len)
                
                weights_end = node_data_end + rank_graph_len
                weights_block = rank_data[node_data_end : weights_end]
                edges_block = rank_data[weights_end:].reshape(-1, 3).astype(int)
                
                node_map = {}

                for node_index in range(rank_graph_len):
                    shell_residues = []
                    shell_solvents = []
                    outer_shell = {}
                    for shell_index in range(num_shells):
                        shell_data = node_block[node_index, shell_index]

                        shell_residues.append(dict(zip(res_keys_sorted, shell_data[:res_len])))
                        shell_solvents.append(dict(zip(solvent_keys_sorted, shell_data[res_len:])))

                    if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
                        outer_shell_data = node_block[node_index, num_shells]
                        outer_shell = dict(zip(all_res_keys_sorted, outer_shell_data))

                    newNode = self.ALL_RANKS_GRAPH.rxnNode(self.CONFIG,shell_residues, shell_solvents, outer_shell)

                    newNodeWeight = weights_block[node_index]

                    map_index = self.ALL_RANKS_GRAPH.updateGraphFromExistingNode(newNode,newNodeWeight)
                    node_map[node_index] = map_index

                for node_index1,node_index2,weight in edges_block:
                    node_index1_mapped = node_map[node_index1]
                    node_index2_mapped = node_map[node_index2]

                    self.ALL_RANKS_GRAPH.updateGraphFromExistingEdgeIndexes(node_index1_mapped,node_index2_mapped,weight)
                    
                current += rank_data_size

    def debugCoordinationShells(self,reactant):
        outfile = open(f"./{self.CONFIG.config['WRITE_DIRECTORY']}/{self.CONFIG.config['RUN_NAME']}/{reactant.index}_{self.STEP}_debug.txt", "w")

        outfile.write(f"{self.clusterFileHeader(reactant)}\n")

        cluster_atoms = np.array([atom_index for mol_index in reactant.total_shells for atom_index in self.MOLECULES[mol_index].atoms])
        cluster_types = set(self.ATOMTYPES[list(cluster_atoms)])

        for shell_index, shell in enumerate([{reactant.index}] + reactant.solvation_shells):
            write = ''
            shell_atoms_by_type = {}
            outfile.write(f"[Shell {shell_index}]\n")
            outfile.write('[Indexes]: index')
            for molecule_index in shell:
                molecule = self.MOLECULES[molecule_index]
                for atom in molecule.atoms:
                    atom_type = self.ATOMTYPES[atom]
                    if atom_type not in shell_atoms_by_type:
                        shell_atoms_by_type[atom_type] = []
                    shell_atoms_by_type[atom_type].append(atom)
                    write += f" {atom}"
            outfile.write(f"{write}\n")

            outfile.write('[VMD Coordination]: ')
            
            write = ''
            for type_key,atom_list in shell_atoms_by_type.items():
                if type_key in self.CUTOFF_TARGETS:
                    for target_t in self.CUTOFF_TARGETS[type_key]:
                        if target_t in cluster_types:
                            write += f"(type {target_t} and within {self.CONFIG.config['PAIR_CUTOFFS'][(type_key,target_t)]} of index {' '.join(map(str,atom_list))}) or "
            outfile.write(write[:-4])
            outfile.write('\n')

    #Graph Analysis Functions -----------------------

    def readGraphFile(self,graph_file_name):

        def extractResidues(file):
            LIGAND_PATTERN = r'(\w+):[\d.]+'
            SOLVENT_PATTERN = r'(\w+)\([\d.]+\)'

            unique_ligands = set()
            unique_solvents = set()
            
            is_data_section = False

            with open(file, 'r') as f:
                for line in f:
                    if "Residue syntax:" in line or "Non-coarsened" in line:
                        continue
                    
                    if "RESIDUE" in line:
                        continue

                    ligands = re.findall(LIGAND_PATTERN, line)
                    unique_ligands.update(ligands)

                    solvents = re.findall(SOLVENT_PATTERN, line)
                    unique_solvents.update(solvents)

            unique_solvents -= unique_ligands

            return unique_ligands, unique_solvents

        def parseDataBlocks(header):
            data_blocks = [0]

            block = 1
            for character in header[1:]:
                if character == '[':
                    if len(data_blocks) > 0:
                        data_blocks.append(block+data_blocks[-1])
                    else:
                        data_blocks.append(block)
                    block = 1
                else:
                    block += 1
            data_blocks.append(block-1+data_blocks[-1])

            return data_blocks
        
        #for graph_file_name in graph_file_names:

        try:
            graph_file = open(graph_file_name, 'r')
        except FileNotFoundError:
            raise configurationError(f"Error: File not found: {graph_file_name}")

        if self.RANK == 0:
            self.CONFIG.STDOUT.write(f"Reading reaction graph file ...\n")

        ligands,solvents = extractResidues(graph_file_name)
        all_residues = ligands | solvents
        self.CONFIG.config['RESIDUE_LIST'] = {res:[] for res in all_residues}
        self.checkConfiguration('COORD_NUM_RESIDUE')
        self.checkConfiguration('K_LIGAND')
        self.checkConfiguration('K_LIGAND_CONC')
        self.checkConfiguration('VIS_SUB_GRAPH_SIZE')
        self.CONFIG.config['COARSEN_SOLVENT_RESIDUES'] = solvents

        lines = graph_file.readlines()

        line_index = 0
        file_len = len(lines)

        while 'REACTION NODES' not in lines[line_index]:
            if line_index+1 == file_len:
                raise mySystemError(f"Unable to parse node section from reaction graph file")
            line_index += 1
        line_index += 9
        nodes_header = lines[line_index]
        if '[Shell 1]' not in nodes_header and '[Probability]' not in nodes_header:
            raise mySystemError(f"Unable to parse node section from reaction graph file")
        
        data_blocks = parseDataBlocks(nodes_header)
        num_shells = len(data_blocks) - 2
        self.CONFIG.config['REACTION_SHELLS'] = num_shells
        if '[Outer shell]' in nodes_header:
            self.CONFIG.config['CONSERVE_COORDINATION'] = True
            self.CONFIG.config['REACTION_SHELLS'] = len(data_blocks) - 3

        line_index += 2

        while line_index < file_len and 'REACTION EDGES' not in lines[line_index]:
            line = lines[line_index]

            if line.strip() == '':
                line_index += 1
                continue

            shell_list = [line[block_width:data_blocks[index+1]] if index < num_shells else line[block_width:] for index,block_width in enumerate(data_blocks[:-1])]

            node = self.GRAPH.rxnNode(self.CONFIG,nodeStringList=shell_list)
            
            self.GRAPH.updateGraphFromExistingNode(newNode=node)

            if line_index == file_len:
                raise mySystemError(f"Unable to parse edge section from reaction graph file")
            
            line_index += 1

        if line_index < file_len:
            line_index += 9
            edges_header = lines[line_index]
            if '[Shell 1]' not in lines[line_index] and '[Probability]' not in edges_header:
                raise mySystemError(f"Unable to parse edge section from reaction graph file")

            WEIGHT_PATTERN = r'\((\d+)\)'
            data_blocks = parseDataBlocks(edges_header)
            edge_num_shells = len(data_blocks) - 2

            line_index += 2

            while line_index < file_len:
                line = lines[line_index]

                if line.strip() == '':
                    line_index += 1
                    continue

                shell_list = [line[block_width:data_blocks[index+1]] if index < edge_num_shells else line[block_width:] for index,block_width in enumerate(data_blocks[:-1])]
                weight_block = '0.0001 (1)' #Dummy weight block only used to instantiate new node
                if len(shell_list) % 2 != 1:
                    raise mySystemError(f"Unable to parse node weight section of reaction graph file line: {''.join(shell_list)}")
                half = len(shell_list)//2
                node1_list = shell_list[:half] + [weight_block]
                node2_list = shell_list[half:-1] + [weight_block]

                weight_match = re.findall(WEIGHT_PATTERN, shell_list[-1])
                if len(weight_match) > 1:
                    raise mySystemError(f"Unable to parse node weight section of reaction graph file line: {''.join(shell_list)}")
                edge_weight = int(weight_match[0])

                node1 = self.GRAPH.rxnNode(self.CONFIG,nodeStringList=node1_list)
                node2 = self.GRAPH.rxnNode(self.CONFIG,nodeStringList=node2_list)
                self.GRAPH.updateGraphFromExistingEdge(From=node1,To=node2,weight=edge_weight)

                line_index += 1

        if self.RANK == 0:
            self.CONFIG.STDOUT.write('\n')

    def graphAnalysis(self):
        
        if self.CONFIG.configExists('COORD_NUM_RESIDUE'):
            coord_numbers = self.GRAPH.coordinationNumber()

            for res in self.CONFIG.config['COORD_NUM_RESIDUE']:
                coord_string = ' '.join([f"Shell {shell_index+1}: {round(shell[res],4)}" for shell_index,shell in enumerate(coord_numbers)])
                if self.RANK == 0:
                    self.CONFIG.STDOUT.write("------Coordination Numbers------\n")
                    self.CONFIG.STDOUT.write(f'{res}: {coord_string}\n\n')

        if self.CONFIG.configExists('K_LIGAND'):
            residues_Ks = self.GRAPH.bindingProbability()

            for res_index in range(len(self.CONFIG.config['K_LIGAND'])):
                res = self.CONFIG.config['K_LIGAND'][res_index]
                concentration = self.CONFIG.config['K_LIGAND_CONC'][res_index]
                if residues_Ks[res] is None:
                    Ks_string = 'K_1: 0'
                else:
                    Ks_string = ' '.join([f"K_{index+1}: {round(constant/concentration,4)}" for index,constant in enumerate(residues_Ks[res])])
                if self.RANK == 0:
                    self.CONFIG.STDOUT.write("------Association Constants------\n")
                    self.CONFIG.STDOUT.write(f'{res}: {Ks_string}\n\n')

        if self.CONFIG.configExists('VIS_SUB_GRAPH_SIZE'):
            self.GRAPH.visuallizeGraph(self.CONFIG.config['VIS_SUB_GRAPH_SIZE'])

class rxnGraph():
    def __init__(self, configuration:"configuration"):

        self.CONFIG = configuration

        self.rxnGraphDict = {} #Stores node indexes under unique node keys (e.g. key:"Shell 1: H2O:5 Shell 2: H2O:10" / value:16) for quick finding + referencing of reaction nodes

        self.NODES = [] #List containing all nodes ordered by node index for quick referencing
        self.graphLen = 0 #current number of nodes in graph 

        self.nodeWeights = [] #key: reactant node index [int] value: weight [int]
        self.edgeWeights = {} #key: (reactant node index [int], product node index [int]) value: weight [int]
        
    #Fundimental object to graph class
    class rxnNode():
        def __init__(self, configuration:"configuration", solvation_shells:list=[], sol_counts:list=None, outer_shell:dict=None, nodeStringList:list=None):
            self.CONFIG = configuration
            self.index = -1
            self.weight = 0
            self.num_solvation_shells = self.CONFIG.config['REACTION_SHELLS']
            self.shells = [{} for _ in range(self.num_solvation_shells)]
            self.solvent_counts = [{} for _ in range(self.num_solvation_shells)]
            self.outer_shell_counts = None
            self.hbond_count = 0

            if nodeStringList:
                #self.shells,self.solvent_counts,self.outer_shell_counts,self.weight = self.str_list_to_node(nodeStringList)
                self.weight,solvation_shells,sol_counts,outer_shell = self.str_list_to_node(nodeStringList)

            self.RESIDUES = sorted(self.CONFIG.config['RESIDUE_LIST'].keys())
            self.COARSENED_RESIDUES = sorted(self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']) if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES') else set()
            self.NON_COARSENED_RESIDUES = sorted(set(self.RESIDUES) - set(self.COARSENED_RESIDUES))

            #solvation_shells and sol_counts should be loaded like this, below is messy
            self.outer_shell_counts = {residue:0 for residue in self.RESIDUES}
            if outer_shell:
                self.updateOuterShell(outer_shell)
            self.products = set()

            if not isinstance(solvation_shells,list) and not all(isinstance(shell,dict) for shell in solvation_shells):
                raise mySystemError("Argument error in rxnNode(): Non-dictionary 'solvation_shells' list object provided to rxnNode(), the correct argument type is [{RES1:RES_NUM1,RES2:RES_NUM2,...},{RES1:RES_NUM1,...},...]")
            if len(solvation_shells) > self.num_solvation_shells:
                self.CONFIG.STDOUT.write('Warning: Argument error in rxnNode(): input solvation_shells oversized, truncating\n')
                solvation_shells = solvation_shells[:self.num_solvation_shells]
            if len(solvation_shells) < self.num_solvation_shells:
                solvation_shells.extend([{} for _ in range(self.num_solvation_shells - len(solvation_shells))])

            for shell in solvation_shells:
                
                missing_defaults = self.RESIDUES - shell.keys()
                invalid_residues = shell.keys() - self.RESIDUES

                if invalid_residues:
                    for res in invalid_residues:
                        self.CONFIG.STDOUT.write("Warning: Argument error in rxnNode(): Invalid dictionary residue key given, removing\n")
                        shell.pop(res)
                
                if missing_defaults:
                    for res in missing_defaults:
                        shell.setdefault(res,0)

            # Solvent Corsening Section:
            # Shift corsened residue data into ranged data struct (most likely solvent type).
            # Residues removed from 'shells' and therefore no longer part of node identity
            # solvent_range holds the node-sepcific range of solvent number encountered in each shell
            #-------------------------------------------
            
            if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
                for shell_index in range(self.num_solvation_shells):
                    for res in self.RESIDUES:
                        if res in self.COARSENED_RESIDUES:
                            self.solvent_counts[shell_index][res] = solvation_shells[shell_index][res]
                        else:
                            self.shells[shell_index][res] = solvation_shells[shell_index][res]

                if sol_counts:
                    if not all(isinstance(shell,dict) for shell in sol_counts):
                        raise mySystemError("Argument error in rxnNode(): Non-dictionary 'sol_counts' list object provided to rxnNode(), the correct argument type is [{SOL1:SOL_NUM,SOL2:SOL_NUM,...},{SOL1:SOL_NUM,...},...]")
                    if len(sol_counts) > self.num_solvation_shells:
                        self.CONFIG.STDOUT.write('Warning: Argument error in rxnNode(): input sol_counts oversized, truncating\n')
                        sol_counts = sol_counts[:self.num_solvation_shells]
                    if len(sol_counts) < self.num_solvation_shells:
                        sol_counts.extend([{} for _ in range(self.num_solvation_shells - len(sol_counts))])

                    for shell in sol_counts:
                        missing_defaults = self.COARSENED_RESIDUES - shell.keys()
                        invalid_residues = shell.keys() - self.COARSENED_RESIDUES

                        if invalid_residues:
                            self.CONFIG.STDOUT.write("Warning: rxnNode() argument error. Invalid sol_counts residue(s) given, {invalid_residues}, removing\n")
                            for res in invalid_residues:
                                shell.pop(res)
                        
                        if missing_defaults:
                            for res in missing_defaults:
                                shell.setdefault(res,0)

                    for shell_index in range(self.num_solvation_shells):
                        self.solvent_counts[shell_index].update(sol_counts[shell_index])
            else:
                self.shells = [{res:count for res,count in shell.items()} for shell in solvation_shells]

        def updateOuterShell(self,outerShell):
            if not (outerShell.keys() <= self.outer_shell_counts.keys()):
                raise mySystemError(f"Attempted to update rxnNode {str(self)} outer shell averages with at least 1 unknown residue: {outerShell.keys()-self.outer_shell_counts.keys()}")

            for residue,count in outerShell.items():
                if residue not in self.outer_shell_counts:
                    self.outer_shell_counts[residue] = count
                else:
                    self.outer_shell_counts[residue] += count

        def updateHBonds(self,hbond_num):
            self.hbond_count += hbond_num

        def addShell(self,contents:dict,shell_index=-1):

            if shell_index < len(self.shells):
                self.CONFIG.STDOUT.write("rxnNode.addShell() ERROR: Invalid index provided\n")
                return
            elif shell_index == -1:
                self.shells.append({})
            
            for key in contents:
                if key not in self.CONFIG.config['RESIDUE_LIST']:
                    self.CONFIG.STDOUT.write("Argument error in rxnNode.editShell: Invalid residue key given\n")
                else:
                    self.shells[shell_index][key] = contents[key]

        def __eq__(self,other):
            if not isinstance(other,rxnGraph.rxnNode):
                return False
            return self.shells == other.shells

        def shell_strings(self):
            #Returns a string (unique for each unique node) containing the residue numbers (and solvent averages if enabled) in each shell
            if self.weight == 0:
                return 'NA'
            shell_string = [f"{' '.join([f'{res}:{shell[res]}' for res in self.NON_COARSENED_RESIDUES if shell[res] != 0] + [f'{res}({round(sol_average/self.weight,2)})' for res, sol_average in self.solvent_counts[index].items()])}" for index, shell in enumerate(self.shells)]
            if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                shell_string += [f"{' '.join([f'{res}({round(self.outer_shell_counts[res]/self.weight,2)})' for res in sorted(self.CONFIG.config['RESIDUE_LIST'])])}"]
            if self.CONFIG.configExists('HBOND_TARGET'):
                shell_string += [f"{round(self.hbond_count/self.weight,2)}"]
            return shell_string

        def __str__(self):
            #Returns a string (unique for each unique node) containing the residue numbers (and solvent averages if enabled) in each shell
            return ' '.join([f"Shell {index + 1}: {' '.join([f'{res}:{shell[res]}' for res in self.NON_COARSENED_RESIDUES if shell[res] != 0])}" for index, shell in enumerate(self.shells)]) #temp
            if self.weight == 0:
                return ' '.join([f"Shell {index + 1}: {' '.join([f'{res}:{shell[res]}' for res in self.NON_COARSENED_RESIDUES if shell[res] != 0])}" for index, shell in enumerate(self.shells)])
            return ' '.join([f"[Shell {index + 1}] {' '.join([f'{res}:{shell[res]}' for res in self.NON_COARSENED_RESIDUES if shell[res] != 0] + [f'{res}({round(sol_average/self.weight,2)})' for res, sol_average in self.solvent_counts[index].items()])}" for index, shell in enumerate(self.shells)])

        def key(self):
            #Returns a string key (unique for each unique node) containing the non-coarsened residue numbers in each shell, used to map node types to their memory locations (sort of like a hash map)
            return ' '.join([f"[Shell {index + 1}] {' '.join([f'{res}:{shell[res]}' for res in self.NON_COARSENED_RESIDUES if shell[res] != 0])}" for index, shell in enumerate(self.shells)])

        def str_list_to_node(self,string_list:int):

            LIGAND_PATTERN = r'(\w+):([\d.]+)'
            SOLVENT_PATTERN = r'(\w+)\(([\d.]+)\)'
            WEIGHT_PATTERN = r'\((\d+)\)'

            shell_contents = []
            solvent_counts = []

            weight_match = re.findall(WEIGHT_PATTERN, string_list[-1]) #Extract node weight from last column
            if len(weight_match) > 1:
                raise mySystemError(f"Unable to parse node weight section of reaction graph file line: {''.join(string_list)}")
            node_weight = int(weight_match[0])
            
            for shell in string_list[:self.CONFIG.config['REACTION_SHELLS']]:
                ligand_matches = re.findall(LIGAND_PATTERN, shell)
                solvent_matches = re.findall(SOLVENT_PATTERN, shell)

                ligand_dict = {k: int(v) for k, v in ligand_matches}
                shell_contents.append(ligand_dict)
                solvent_dict = {k: int(round(float(v) * node_weight)) for k, v in solvent_matches}
                solvent_counts.append(solvent_dict)

            outer_counts = None
            if self.CONFIG.config['CONSERVE_COORDINATION'] is not None:
                outer_matches = re.findall(SOLVENT_PATTERN, string_list[-2])
                outer_counts = {k: int(round(float(v) * node_weight)) for k, v in outer_matches}

            return node_weight,shell_contents,solvent_counts,outer_counts

    def getSize(self):
        return self.graphLen

    def updateGraph(self, temp, current):
        
        found = None

        #Algorithm for updating rxn graph while multiple species are traversing

        if not current: #First recorded coordination state of species 'molecule'

            found = self.indexFind(temp)

            if found: #If matching node is found set 'molecule' 's current pointer to that node
                self.updateSolventRange(newNode=temp,oldNode=found)
                current = found
            else: #If matching node is not found in the current rxn graph add a new root to NODES list   
                self.storeNewNode(temp)
                current = temp
        elif temp == current:
            self.updateSolventRange(newNode=temp,oldNode=current)
        else:
            previous = current

            found = self.indexFind(temp)

            if found:
                self.updateSolventRange(newNode=temp,oldNode=found)
                #update products (edge) list
                if found.index not in current.products:
                    current.products.add(found.index)
                current = found
            else:
                self.storeNewNode(temp)
                current.products.add(temp.index)
                current = temp

            self.updateWeights(previous,current)

        self.updateWeights(current)

        return current
    
    def updateGraphFromExistingNode(self,newNode:rxnNode,newNodeWeight:int=None):

        if newNodeWeight is None:
            if newNode.weight == 0:
                raise mySystemError("Attempted to add unvisited node to graph")
            newNodeWeight = newNode.weight

        found = self.indexFind(newNode)

        if found is None:
            self.storeNewNode(newNode)
            newNode.weight = newNodeWeight
            self.nodeWeights.append(newNodeWeight)
            return newNode.index
        
        self.updateSolventRange(newNode,found,newNodeWeight)
        found.updateOuterShell(newNode.outer_shell_counts)
        found.weight += newNodeWeight
        self.nodeWeights[found.index] += newNodeWeight
        return found.index

    def updateGraphFromExistingEdge(self,From:rxnNode,To:rxnNode,weight:int):

        foundFrom = self.indexFind(From)
        foundTo = self.indexFind(To)
        indexFrom = foundFrom.index
        indexTo = foundTo.index
        foundFrom.products.add(indexTo)
        edge_key = (indexFrom,indexTo)
        if edge_key in self.edgeWeights:
            self.edgeWeights[edge_key] += weight
        else:
            self.edgeWeights[edge_key] = weight

    def updateGraphFromExistingEdgeIndexes(self,indexFrom:int,indexTo:int,weight):
        self.NODES[indexFrom].products.add(indexTo)
        edge_key = (indexFrom,indexTo)
        if edge_key in self.edgeWeights:
            self.edgeWeights[edge_key] += weight
        else:
            self.edgeWeights[edge_key] = weight

    def updateSolventRange(self,newNode:rxnNode,oldNode:rxnNode,newNodeWeight:int=None):
        if not self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
            return

        for shell_index in range(self.CONFIG.config['REACTION_SHELLS']):
            for solvent_key in self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']:
                oldNode.solvent_counts[shell_index][solvent_key] += newNode.solvent_counts[shell_index][solvent_key]

    def addEdge(self,From:rxnNode,To:rxnNode,bidirectional:bool=False):
        
        foundFrom = self.indexFind(From)

        foundTo = self.indexFind(To)

        if not foundFrom:
            foundFrom = From
            self.storeNewNode(foundFrom)

        if not foundTo:
            foundTo = To
            self.storeNewNode(foundTo)

        self.updateSolventRange(newNode=To,oldNode=foundTo)
        self.updateSolventRange(newNode=From,oldNode=foundFrom)

        if foundTo.index not in foundFrom.products:
            foundFrom.products.add(foundTo.index)
        if bidirectional and foundFrom.index not in foundTo.products:
            foundTo.products.append(foundFrom.index)

    def updateWeights(self,reactantNode:rxnNode,productNode:rxnNode=None):

        if not productNode:
            if reactantNode.index >= len(self.nodeWeights):
                reactantNode.weight = 1
                self.nodeWeights.append(1)
            else:
                reactantNode.weight += 1
                self.nodeWeights[reactantNode.index] += 1
        else:
            if (reactantNode.index,productNode.index) not in self.edgeWeights:
                self.edgeWeights[(reactantNode.index,productNode.index)] = 1
            else:
                self.edgeWeights[(reactantNode.index,productNode.index)] += 1

    def writeWeights(self,out,start,end):

        if not self.nodeWeights and not self.edgeWeights:
            return
        
        gap_width = 4
        out.write(f'REACTION GRAPH for FRAMES: {start} - {end}\n\n')
        
        if self.nodeWeights:
            out.write('--------REACTION NODES--------\n\nSyntax:\n[Node Contents e.g. [Shell 1] [Shell 2] ... ] [Probability]\n\nResidue syntax:\nCoarsened residues- RESIDUE(avg)\nNon-coarsened residues- RESIDUE:count\n\n')

            total_weight = sum(self.nodeWeights)
            sorted_indices = sorted(range(len(self.nodeWeights)), key=lambda k: self.nodeWeights[k], reverse=True)
            sorted_weights = [self.nodeWeights[index] for index in sorted_indices]
            sorted_nodes = [self.NODES[index].shell_strings() for index in sorted_indices]
            sorted_probabilities = [round((self.nodeWeights[index] / total_weight),4) for index in sorted_indices]
            max_lengths = [len(max(column, key=len))+gap_width for column in zip(*sorted_nodes)]
            
            headers = []
            shell_header_len = int(len(max_lengths))
            formatted_header = []
            if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                headers.append('[Outer shell]')
                shell_header_len -= 1
            if self.CONFIG.configExists('HBOND_TARGET'):
                headers.append('[Avg. H-Bonds]')
                shell_header_len -= 1
            for i in range(shell_header_len-1,-1,-1):
                headers.insert(0,f"[Shell {i+1}]")
            for i, length_header in enumerate(zip(max_lengths,headers)):
                content_length = length_header[0]
                header = length_header[1]
                header_length = len(header) + gap_width
                if header_length > content_length: max_lengths[i] = content_length = header_length
                formatted_header.append(f"{header:<{content_length}}")
            formatted_header.append('[Probability]\n\n')
            out.write(''.join(formatted_header))

            out.write('\n'.join([''.join([f"{sorted_nodes[i][j]:<{length}}" for j,length in enumerate(max_lengths)] + [f"{sorted_probabilities[i]} ({sorted_weights[i]})"]) for i in range(self.graphLen)]))

        if self.edgeWeights:
            #write edge weights
            out.write('\n\n--------REACTION EDGES--------\n\nSyntax:\n[Reactant Node Contents e.g. [Shell 1] [Shell 2] ... ] [Product Node Contents e.g. [Shell 1] [Shell 2] ... ] [Probability]\n\nResidue syntax:\nCoarsened residues- RESIDUE(avg)\nNon-coarsened residues- RESIDUE:count\n\n')

            total_weight = sum(weight for weight in self.edgeWeights.values())
            sorted_weights = sorted(self.edgeWeights.items(), key=lambda item: item[1], reverse=True)
            sorted_edges = [self.NODES[item[0]].shell_strings() + self.NODES[item[1]].shell_strings() for item, weight in sorted_weights]
            sorted_probabilities = [round((abs(weight) / total_weight),4) for item, weight in sorted_weights]
            max_lengths = [len(max(column, key=len))+gap_width for column in zip(*sorted_edges)]
            
            headers = []
            total_header_len = int(len(max_lengths)/2)
            shell_header_len = total_header_len
            formatted = []
            if self.CONFIG.configExists('CONSERVE_COORDINATION'):
                headers.append('[Outer shell]')
                shell_header_len -= 1
            if self.CONFIG.configExists('HBOND_TARGET'):
                headers.append('[Avg. H-Bonds]')
                shell_header_len -= 1
            for i in range(shell_header_len-1,-1,-1):
                headers.insert(0,f"[Shell {i+1}]")
            formatted_node1 = []
            formatted_node2 = []
            for i in range(total_header_len):
                j = i + total_header_len
                header = headers[i]
                header_length = len(header) + gap_width
                content_length_1 = max_lengths[i]
                content_length_2 = max_lengths[j]
                if header_length > content_length_1: max_lengths[i] = content_length_1 = header_length
                if header_length > content_length_2: max_lengths[j] = content_length_2 = header_length
                formatted_node1.append(f"{header:<{content_length_1}}")
                formatted_node2.append(f"{header:<{content_length_2}}")

            formatted = formatted_node1 + formatted_node2
            formatted.append('[Probability]\n\n')
            out.write(''.join(formatted))

            out.write('\n'.join([''.join([f"{sorted_edges[i][j]:<{length}}" for j,length in enumerate(max_lengths)] + [f"{sorted_probabilities[i]} ({sorted_weights[i][1]})"]) for i in range(len(self.edgeWeights))]))

    def storeNewNode(self,newNode:rxnNode):

        newNode.index = self.graphLen #Update node index

        self.rxnGraphDict[newNode.key()] = newNode.index #Place an index entry into the rxnNode dictionary

        #This should probably be removed after testing to avoid expense
        if newNode in self.NODES:
            raise Exception("rxnGraph.storeNewNode() attempted to add duplicate to NODES")
        else:
            self.NODES.append(newNode) #Add new node to list

        self.graphLen += 1 #Update the length of the graph

    def indexFind(self,find:rxnNode):

        nodeKey = find.key()

        if nodeKey in self.rxnGraphDict:
            return self.NODES[self.rxnGraphDict[nodeKey]]
            
        return None

    def graphFind(self, root, find, visited=None): #root = rxn graph root node, find = node to find in graph, visited = visited set to prevent double checks, found = found assignment if node is found 
        
        #DFS graph searching algorithm

        if visited == None: #Visited data struct for graph searching, stores node indexes in set cause its faster than storing nodes themselves in a list
            visited = set()

        if root == find:
            return root
        
        visited.add(root.index)

        for product in root.products:
            if product.index not in visited:
                found = self.graphFind(product,find,visited)

                if found:
                    return found
            
        return None

    def writeRxnGraph(self, out, time):

        out.write('----REACTION GRAPH for TIMESTEP: ' + str(time) + '----' + '\n') #if print has just started, print header to seperate from previous print statements

        visited = set()

        write = ''
        for node in self.NODES:
            visited.add(node.index)
            for product_index in node.products:
                product = self.NODES[product_index]
                if node.index in product.products: #Two-way graph edge found, instead of writing both edges (node1-->node2 ... node2-->node1), write as '<-->'
                    if product.index in visited:
                        continue
                    write += ''.join([str(node),' <--> ',str(product),'\n'])
                else:
                    write += ''.join([str(node),' --> ',str(product),'\n'])
        out.write(write)

    def serialize(self,rank):

        graph_dtype = np.int64

        if self.graphLen == 0:
            return np.empty(shape=(0,), dtype=graph_dtype)

        all_res_keys_sorted = sorted(set(self.CONFIG.config['RESIDUE_LIST'].keys()))
        res_keys_sorted = sorted(set(self.CONFIG.config['RESIDUE_LIST'].keys()) - (set() if not self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES') else set(self.CONFIG.config['COARSEN_SOLVENT_RESIDUES'])))
        solvent_keys_sorted = sorted((set() if not self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES') else self.CONFIG.config['COARSEN_SOLVENT_RESIDUES']))

        all_res_len = len(all_res_keys_sorted)
        res_len = len(res_keys_sorted)
        sol_len = len(solvent_keys_sorted)
        shell_len = res_len + sol_len
        if all_res_len != shell_len:
            raise mySystemError('Graph unpacking chunk size issue, check residue list')

        num_shells = self.CONFIG.config['REACTION_SHELLS']
        edge_len = len(self.edgeWeights)
        data_length = (self.graphLen * num_shells * shell_len) + self.graphLen + (edge_len * 3)
        if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
            data_length += self.graphLen * shell_len

        graph_data = np.empty(data_length, dtype=graph_dtype)
        #graph_data = np.empty(data_length, dtype=int)

        current = 0
        get_all_res = (itemgetter(*all_res_keys_sorted) if all_res_len > 0 else None)
        get_res = (itemgetter(*res_keys_sorted) if res_len > 0 else None)
        get_sol = (itemgetter(*solvent_keys_sorted) if sol_len > 0 else None) 

        for node in self.NODES:
            for shell, sol in zip(node.shells, node.solvent_counts):
                if res_len > 0:
                    graph_data[current:current+res_len] = get_res(shell)
                current += res_len
                if sol_len > 0:
                    graph_data[current:current+sol_len] = get_sol(sol)
                current += sol_len
            
            if self.CONFIG.configExists('COARSEN_SOLVENT_RESIDUES'):
                graph_data[current:current+shell_len] = get_all_res(node.outer_shell_counts)
                current += shell_len

        graph_data[current:(current + self.graphLen)] = self.nodeWeights
        current += self.graphLen

        for (u, v), w in self.edgeWeights.items():
            graph_data[current] = u
            graph_data[current+1] = v
            graph_data[current+2] = w
            current += 3

        return graph_data

    def coordinationNumber(self):
        total_node_weights = sum([node.weight for node in self.NODES])

        shell_len = len(self.NODES[0].shells)

        coord_numbers = [{} for shell in range(shell_len)]
        for node in self.NODES:
            for shell_index,shell in enumerate(node.shells):
                for res in shell:
                    if res in coord_numbers[shell_index]:
                        coord_numbers[shell_index][res] += shell[res] * (node.weight / total_node_weights)
                    else:
                        coord_numbers[shell_index][res] = shell[res] * (node.weight / total_node_weights)

        return coord_numbers

    def bindingProbability(self):

        #Returns a dictionary containing the 1 - N order bound/unbound probability ratios for a given ligand to the reactant. E.g. {Res_type_1:(0.133,0.04),... }
        #1st order = p(RL)/p(R)
        #2nd order = p(RLL)/p(RL)

        residue_ratios = {}

        for node in self.NODES:
            for res,count in node.shells[0].items(): #Explore only 1st shell binding
                if res not in residue_ratios:
                    residue_ratios[res] = []
                
                residue_ratios[res].extend([0 for _ in range(count+1-len(residue_ratios[res]))])
                
                residue_ratios[res][count] += node.weight

        return {res: [residue_ratios[res][n] / residue_ratios[res][n-1] if residue_ratios[res][n] != 0 and residue_ratios[res][n-1] != 0 else 0 for n in range(1, len(ratios))] if len(ratios) > 1 else None for res, ratios in residue_ratios.items()}

    def visuallizeGraph(self,size):

        import networkx as nx
        import matplotlib.pyplot as plt

        node_indexes = []
        nodes = []
        nodes_total_weight = 0
        edges = []
        edges_total_weight = 0

        #Sort and filter nodes by weight
        sorted_indices = sorted(range(len(self.nodeWeights)), key=lambda k: self.nodeWeights[k], reverse=True)
        sorted_node_weights = [self.nodeWeights[index] for index in sorted_indices]

        #Fix!!
        # for node_index,weight in zip(sorted_indices[:size],sorted_node_weights[:size]):
        #     node_indexes.append(node_index)
        #     #nodes.append((str(self.NODES[node_index]),{'weight':weight}))
        #     nodes.append((node_index,{'weight':weight}))
        #     nodes_total_weight += weight

        for index in size:
            node_index = sorted_indices[index]
            weight = sorted_node_weights[node_index]
            node_indexes.append(node_index)
            nodes.append((node_index,{'weight':weight}))
            nodes_total_weight += weight

        for node in nodes:
            node[1]['weight'] = node[1]['weight'] / nodes_total_weight

        #Sort and filter edges by weight
        #sorted_edge_weights = sorted(self.edgeWeights.items(),key=lambda item: item[1],reverse=True)

        for node1 in node_indexes:
            for node2 in node_indexes:
                if node1 == node2:
                    continue
                edge_key = (node1,node2)
                if edge_key in self.edgeWeights:
                    weight = self.edgeWeights[edge_key]
                    edges_total_weight += weight
                    #edges.append((str(self.NODES[node1]),str(self.NODES[node2]),weight))
                    edges.append((node1,node2,weight))

        for index,edge in enumerate(edges):
            edges[index] = (edge[0],edge[1],(edge[2] / edges_total_weight))

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        node_labels = {
            n: f"{n}\n({G.nodes[n].get('weight', 0):.3f})" 
            for n in G.nodes()
        }

        # 2. Extract edge weights into a clean dictionary format for NetworkX
        edge_labels = {
            (u, v): f"{G[u][v]['weight']:.4f}" 
            for u, v in G.edges()
        }

        # Extraction for positioning
        node_names = [n[0] for n in nodes]
        pos = nx.circular_layout(node_names,scale=0.5) 

        node_sizes = [G.nodes[n].get('weight', 0.1) * 50000 for n in G.nodes()]
        edge_widths = [((G[i][j]['weight'] * 40) if G[i][j]['weight'] > 0.002 else 0.08) for i, j in G.edges()]

        plt.figure(figsize=(6, 6))
        
        # Draw the main graph structure (Note: with_labels=False here because we do it next)
        nx.draw(
            G, 
            pos, 
            with_labels=False, 
            node_color="green", 
            node_size=node_sizes,
            width=edge_widths,
            edge_color="black", 
            alpha=1,
            arrowsize=15,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1"
        )

        # Draw our custom node labels (Name + Weight) directly over the nodes
        nx.draw_networkx_labels(
            G, 
            pos, 
            labels=node_labels, 
            font_weight="bold", 
            font_size=10
        )

        # Draw the edge weight labels floating along the lines
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=edge_labels, 
            font_size=9,
            label_pos=0.5,  # 0.5 means exactly halfway down the edge line
            connectionstyle="arc3,rad=0.1"
        )

        plt.gca().margins(0.15)

        plt.title(f"Reaction sub-graph for {size} highest probability nodes")
        plt.show()
        pass

class mySystemError(Exception):
    pass

class configurationError(Exception):
    pass

class configuration():

    def __init__(self,cl_input_file:str=None):

        self.profiler = cProfile.Profile()
        self.COMM = MPI.COMM_WORLD
        self.RANK = self.COMM.Get_rank()

        self.INPUT_FILE_NAME = cl_input_file

        if os.path.isfile(self.INPUT_FILE_NAME):
            self.INPUT_FILE_NAME = os.path.abspath(self.INPUT_FILE_NAME)
        else:
            raise configurationError('Command line input file not found')

        self.STDOUT = None #Write location for standard output
        self.STDERR = None #Write location for standard output
        self.METAOUT = None #Write location for performance analytics
        self.CONFIGOUT = None #Write location for config_ file (run input copy)

        self.config = {
            #Input file reading parameters----------
            'SYSTEM_TYPE':None,
            'DATA_FILE_PATH':None,
            'TRJ_FILE_PATH':None,
            'TOPOLOGY_FILE_PATH':None,
            'DUMP_FREQ':None,
            'FRAMES_TO_PROCESS':-1,

            #Additional atom type / residue type parameters (if missing in data/traj file)----------
            'ATOM_TYPE_LIST':None,
            'RESIDUE_LIST':None,

            #Trajectory analysis parameters----------
            'REACTANT':None,
            'REACTION_SHELLS':2,
            'CLUSTER_MOLECULES':150,
            'PAIR_CUTOFFS':None,

            #Analysis output / cluster extraction parameters----------
            'CREATE_RXN_GRAPH':False,
            'WRITE_DIRECTORY':None,
            'RUN_NAME':None,
            'OUTPUT_TYPE':'XYZ',
            'CLUSTERS_TO_EXTRACT':0,
            'REACTANT_TO_PRINT':None,
            'CONSERVE_COORDINATION':None,
            'COARSEN_SOLVENT_RESIDUES':None,
            'EXCLUDE_SOLVENT_RESIDUES':None,

            #Extraction/reaction graph type filtering control
            'FILTER_REACTANTS_BY_Z':None,
            'HBOND_TARGET':None,
            'HBOND_DEV':None,
            'SPECTATOR_TARGET':None,
            'SPECTATOR_DEV':None,

            #Residue charge /dielectric parameters----------
            'RESIDUE_DIELECTRICS':None,
            'RESIDUE_CHARGES':None,

            #QCHEM input file parameters----------
            'QC_BASIS':None,
            'QC_BASIS2':None,
            'QC_PSEUDO':None,
            'QC_METHOD':None,
            'QC_PCM_METHOD':None,

            #Print frame-by-frame coordination info (for debugging)----------
            'DEBUG':False,
            #Print time profiling statistics----------
            'PROFILE':False,

            #Reaction Graph analysis user variables------------------

            'COORD_NUM_RESIDUE':None,
            'K_LIGAND':None,
            'K_LIGAND_CONC':None,
            'VIS_SUB_GRAPH_SIZE':None
            }

        self.types = {
            #Input file reading parameters----------
            'SYSTEM_TYPE':str,
            'DATA_FILE_PATH':str,
            'TRJ_FILE_PATH':str,
            'TOPOLOGY_FILE_PATH':str,
            'DUMP_FREQ':int,
            'FRAMES_TO_PROCESS':[int,list,tuple],

            #Additional atom type / residue type parameters (if missing in data/traj file)----------
            'ATOM_TYPE_LIST':list,
            'RESIDUE_LIST':dict,

            #Trajectory analysis parameters----------
            'REACTANT':str,
            'REACTION_SHELLS':int,
            'CLUSTER_MOLECULES':int,
            'PAIR_CUTOFFS':dict,

            #Analysis output / cluster extraction parameters----------
            'CREATE_RXN_GRAPH':bool,
            'WRITE_DIRECTORY':str,
            'RUN_NAME':str,
            'OUTPUT_TYPE':str,
            'CLUSTERS_TO_EXTRACT':int,
            'REACTANT_TO_PRINT':list,
            'CONSERVE_COORDINATION':dict,
            'COARSEN_SOLVENT_RESIDUES':list,
            'EXCLUDE_SOLVENT_RESIDUES':list,

            #Extraction/reaction graph type filtering control
            'FILTER_REACTANTS_BY_Z':[tuple,list],
            'HBOND_TARGET':[int,float],
            'HBOND_DEV':[int,float],
            'SPECTATOR_TARGET':dict,
            'SPECTATOR_DEV':dict,

            #Residue charge /dielectric parameters----------
            'RESIDUE_DIELECTRICS':dict,
            'RESIDUE_CHARGES':dict,

            #QCHEM input file parameters----------
            'QC_BASIS':[str,dict],
            'QC_BASIS2':[str,dict],
            'QC_PSEUDO':dict,
            'QC_METHOD':str,
            'QC_PCM_METHOD':str,

            #Print frame-by-frame coordination info (for debugging)----------
            'DEBUG':bool,
            #Print time profiling statistics----------
            'PROFILE':bool,

            #Reaction Graph analysis user variables------------------

            'COORD_NUM_RESIDUE':[str,list],
            'K_LIGAND':[str,list],
            'K_LIGAND_CONC':[int,float,list],
            'VIS_SUB_GRAPH_SIZE':[int,list]
            }

        self.comments = {
            'SYSTEM_TYPE':'Molecular dynamics simulation type',
            'DATA_FILE_PATH':'Relative path to data file',
            'TRJ_FILE_PATH':'Relative path to trajectory file',
            'TOPOLOGY_FILE_PATH':'Optional topology file',
            'DUMP_FREQ':'timestep in fs',
            'FRAMES_TO_PROCESS':'Trajectory frame(s) to process',

            'ATOM_TYPE_LIST':'Ordered atom type list if no atom type are provided',
            'RESIDUE_LIST':'Dictionary matching residue types to un-ordered atom type lists',

            'REACTANT':'Reactant residue type',
            'REACTION_SHELLS':'Number of solvation shells to include in reaction graph and extracted clusters',
            'CLUSTER_MOLECULES':'Number of molecules in each neighbor cluster',
            'PAIR_CUTOFFS':'Atom-atom pair cutoffs to use in determing coordination in angstroms',

            'CREATE_RXN_GRAPH':'Toggle creation of reaction graph',
            'WRITE_DIRECTORY':'Master directory for all sub-runs',
            'RUN_NAME':'Sub-directory for analysis and extraction output',
            'OUTPUT_TYPE':"File type to write extracted clusters to",
            'CLUSTERS_TO_EXTRACT':'Number of clusters to extract from trajectory',
            'REACTANT_TO_PRINT':'Reactant solvation shell composition to extract',
            'CONSERVE_COORDINATION':'Residue(s) and number(s) to conserve for each specie',
            'COARSEN_SOLVENT_RESIDUES':'Residues to corsen when building reaction graph. Will not trace coordination changes from these species, instead storing a range',
            'EXCLUDE_SOLVENT_RESIDUES':"Specifies residues to explicitly exlude from extracted clusters when utilizing solvent corsening, must be a subset of 'COARSEN_SOLVENT_RESIDUES'. Not necessary if utilizing 'CONSERVE_COORDINATION', which exactly specifies the composition of the cluster",

            #Extraction/reaction graph type filtering control
            'FILTER_REACTANTS_BY_Z':'#Z bounds within which reactants must be to be filtered for extraction or reaction graph, must be in angstroms. Multiple bounds may be used',
            'HBOND_TARGET':'Number of H-Bonds that extracted clusters should have',
            'HBOND_DEV':"Maximum allowed deviation (plus or minus) from 'HBOND_TARGET' for extracted clusters",
            'SPECTATOR_TARGET':'Number of each type of spectator (outer shell) ion clusters should contain',
            'SPECTATOR_DEV':"Maximum allowed deviation (plus or minus) from 'SPECTATOR_TARGET' for extracted clusters", 

            'RESIDUE_DIELECTRICS':'Residue specific dielectric constants',
            'RESIDUE_CHARGES':'Residue specific charges',

            'QC_BASIS':'Basis set to use in QCHEM calculation, may be either a single string for a global basis set or a complete dictionary containing atom_type:basis_set key value pairs when using different basis sets for different atom types, NOTE: The basis set for each atom type in QC_PSEUDO which utilizes a pseudopotential will be replaced by its associated basis set unless this keyword is a complete dictionary',
            'QC_BASIS2':'Auxilary basis set to use for QCHEM calculation SCF speed-up, may be either a single string for a global aux basis set or a complete dictionary containing atom_type:aux_basis_set key value pairs when using different aux basis sets for different atom types, NOTE: The aux basis set for each atom type in QC_PSEUDO which utilizes a pseudopotential will be replaced by its associated basis set unless this keyword is a complete dictionary',
            'QC_PSEUDO':'Atom type pseudopotential(s) to use in QCHEM calculation, must be a dictionary containing atom_type:pseudopotential key value pairs, does not have to be complete, NOTE: The basis set (and aux basis) for each atom type in QC_PSEUDO which utilizes a pseudopotential will be replaced by its associated basis set unless QC_BASIS is complete',
            'QC_METHOD':'Method used to solve the schrodinger equation in QCHEM calculation',
            'QC_PCM_METHOD':'Polarizable Continuum Model to use in QCHEM calculation',
                            
            'DEBUG':'Prints coordination data for each reactant for debugging purposes',
            'PROFILE':'Utilize time profiling',

            #Reaction Graph analysis user variables------------------

            'COORD_NUM_RESIDUE':'Residue type (string) or types (list) to find coordination numbers for',
            'K_LIGAND':'Residue type (string) or types (list) to find association constant(s) for (K = [R(K_LIGAND)]/[R][K_LIGAND])',
            'K_LIGAND_CONC':'Concentration (string) or concentrations (list) of K_LIGAND (s) in M (mol/liter), if using list: length must match K_LIGAND',
            'VIS_SUB_GRAPH_SIZE':"Visuallizes a sub graph containing the 'VIS_SUB_GRAPH_SIZE' highest probability nodes in the reaction graph and their connectivity"
            }

        self.syntax = {
            'SYSTEM_TYPE':"'gromacs' or 'lammps'",
            'DATA_FILE_PATH':"",
            'TRJ_FILE_PATH':"",
            'TOPOLOGY_FILE_PATH':"",
            'DUMP_FREQ':"",
            'FRAMES_TO_PROCESS':"frame or (start,stop,step), frame: process first 'frame' frames, -1 --> process all frames, (start,stop,step) process slice of trajectory following python slicing notation (e.g. (10,100,2) --> [10:100:2] iterate trajecory from frame 10 to 100 in steps of 2)",

            'ATOM_TYPE_LIST':'[atom_type_1_name,atom_type_2_name,...] ex. [HW,Li,OW]',
            'RESIDUE_LIST':'{residue_type_1:[atom_type_1,atom_type_2,...],...} e.g. {H2O:[OW,HW,HW],Li:[Li],NO3-:[NO,ON,ON,ON]}',

            'REACTANT':"",
            'REACTION_SHELLS':"",
            'CLUSTER_MOLECULES':"",
            'PAIR_CUTOFFS':'{(atom_type_x,atom_type_y):cutoff,... or ((atom_type_x,atom_type_y,...),atom_type_z):cutoff,... or ((atom_type_x,atom_type_y,...),(atom_type_i,atom_type_j,...)):cutoff,...}  NOTE: Does not need to include all atom pairs',

            'CREATE_RXN_GRAPH':'True or False, default: False',
            'WRITE_DIRECTORY':"",
            'RUN_NAME':"",
            'OUTPUT_TYPE':"options: 'XYZ'- generic structure file, 'QCHEM'- QCHEM compatable input file, 'GRO'- gromacs structure file, for multiple file outputs seperate keywords with '_' e.g., 'GRO_XYZ' ",
            'CLUSTERS_TO_EXTRACT':"",
            'REACTANT_TO_PRINT':'[{shell_1_residue:shell_1_residue_number,...},{shell_2_residue:shell_2_residue_number,...},...] ex. [{H2O:4,NIT:1},{H2O:12,NIT:1}]',
            'CONSERVE_COORDINATION':'{specie_residue_1:{conserved_residue_1:conserved_residue_number_1,...},...} ex. {UYL:{H2O:30},NIT:None,TBP:None}',
            'COARSEN_SOLVENT_RESIDUES':'[residue_type_1,residue_type_2,...]',
            'EXCLUDE_SOLVENT_RESIDUES':'[residue_type_1,residue_type_2,...]',

            #Extraction/reaction graph type filtering control
            'FILTER_REACTANTS_BY_Z':'(z_lower_bound_1, z_upper_bound_1) or (z_lower_bound_1, z_upper_bound_1,z_lower_bound_2, z_upper_bound_2,...) where z_lower_bound_1 < z_upper_bound_1 < z_lower_bound_2 < z_upper_bound_2',
            'HBOND_TARGET':'Number greater than 0',
            'HBOND_DEV':'Number greater than 0',
            'SPECTATOR_TARGET':'{residue_type_1:target,...} e.g. {Li:2,NIT:4,...}',
            'SPECTATOR_DEV':'{residue_type_1:deviation,...} e.g. {Li:1,NIT:2,...}', 

            'RESIDUE_DIELECTRICS':'{residue_type_1:dielectric,...} e.g. {H2O:78.4,HEXA:4,NIT:78.4}',
            'RESIDUE_CHARGES':'{residue_type_1:charge,...} e.g. {H2O:0,HEXA:0,NIT:-1}',

            'QC_BASIS':"'QCHEM_basis_set_keyword' or {atom_type1:basis_set_keyword,atom_type2:basis_set_keyword,...}",
            'QC_BASIS2':"'QCHEM_aux_basis_set_keyword' or {atom_type1:aux_basis_set_keyword,atom_type2:aux_basis_set_keyword,...}",
            'QC_PSEUDO':"{atom_type1:pseudopotential_keyword,atom_type2:pseudopotential_keyword,...}",
            'QC_METHOD':"'QCHEM_method_keyword' e.g., 'HF', 'B3LYP', etc",
            'QC_PCM_METHOD':"",
                            
            'DEBUG':"",
            'PROFILE':"",

            #Reaction Graph analysis user variables------------------

            'COORD_NUM_RESIDUE':"Residue_type or [Residue_type1,Residue_type2,...]",
            'K_LIGAND':"Residue_type or [Residue_type1,Residue_type2,...]",
            'K_LIGAND_CONC':"Residue_concentration or [Residue_type1_concentration,Residue_type2_concentration,...]",
            'VIS_SUB_GRAPH_SIZE':''
            }

        if not (self.config.keys() == self.types.keys() == self.comments.keys() == self.syntax.keys()):
            raise configurationError('configuration class initialization error')

    def configExists(self,parameter_name):
        return not (self.config[parameter_name] is None)

    def loadCommandLine(self):

        if len(sys.argv) < 3:
            raise configurationError("Missing command line input file. Run command is 'python mySystem_mpi.py config_file_name' (for serial) or 'mpiexec -n NUM_CORES python mySystem_mpi.py config_file_name' (for parallel)")
        elif len(sys.argv) > 3:
            if self.RANK == 0:
                self.STDOUT.write('WARNING: Excess command line arguments provided. Arguments beyond command line intput file name will be ignored\n')
        else:
            self.INPUT_FILE_NAME = sys.argv[2]
            if os.path.isfile(self.INPUT_FILE_NAME):
                pass
            elif os.path.isfile(os.path.join(os.getcwd(),self.INPUT_FILE_NAME)):
                self.INPUT_FILE_NAME = os.path.join(os.getcwd,self.INPUT_FILE_NAME)
            else:
                raise configurationError('Command line input file not found')
                
    def readInputFile(self):

        import ast
        from difflib import get_close_matches
        
        configIn = open(self.INPUT_FILE_NAME,'r')

        config_lines = configIn.readlines()
        content_lines = []

        #Remove comments --------------
        for line_index,line in enumerate(config_lines):

            #Skip purely comment lines
            line = line.split('#')
            
            if not line[0].strip():
                continue
            
            #Trim and keep lines with content
            content_lines.append(line[0].strip())
            line = line[0].strip()

        #Use numpy fancy indexing to 'pop' commented lines and strip remaining lines
        config_lines = content_lines

        #Aggregate parameter names and parameters on single and multiple lines --------------
        param_dict = {}
        continue_param = ''

        for line_index,line in enumerate(config_lines):

            if '=' in line: 
                
                eq_delim_split = [part.strip() for part in line.split('=')]

                if len(eq_delim_split) > 2: #Line contains multiple '=' --> exit
                    raise configurationError(f'Reading configuration file, cannot resolve line: "{line}". Parameter lines must begin with "PARAMETER_KEYWORD = " or "PARAMETER_KEYWORD=", the parameter may span multiple lines')
                
                if len(eq_delim_split[0].split()) > 1: #Parameter is not a single word --> exit
                    raise configurationError(f'Reading configuration file, cannot resolve line: "{line}". Parameter lines must begin with "PARAMETER_KEYWORD = " or "PARAMETER_KEYWORD=", the parameter may span multiple lines')

                if not eq_delim_split[0].split(): #Line contains leading '='
                    raise configurationError(f'Reading configuration file, cannot resolve line: "{line}". Parameter lines must begin with "PARAMETER_KEYWORD = " or "PARAMETER_KEYWORD=", the parameter may span multiple lines')

                continue_param = eq_delim_split[0]
                param_dict[eq_delim_split[0]] = eq_delim_split[1]
            else:
                try:
                    param_dict[continue_param] += line
                except IndexError:
                    raise configurationError(f'Reading configuration file, cannot resolve line: "{line}"')

        ATOM_TYPE_REGEX = r'(?<![a-zA-Z0-9+_.\-\/"\'\b])(?!(?:-?\d+\.?\d*|(?:"|\'))(?:[,\]\)\s:}]|$))([a-zA-Z0-9+_.\-\/]+)(?![a-zA-Z0-9+_.\-\/"\'\b])' #Captures non-quoted parameters and puts them in quotes for python literal interpretation, will capture parameters with '.-_+' and numbers, will not capture fully digit parameters

        #Remove whitespace, add quotes to strings, and attempt to read parameters as python literals using ast --------------
        for parameter_name, parameter in param_dict.items():

            if parameter_name not in self.config:
                closest_match = get_close_matches(parameter_name, self.config.keys(), n=1, cutoff=0.6)
                raise configurationError(f"Reading configuration file, invalid parameter keyword: '{parameter_name}'. Did you mean '{closest_match[0]}'")

            #Remove leading, trailing, and interjected white space
            parameter.replace(" ","")
                
            if parameter != 'None' and parameter != 'none' and parameter != 'True' and parameter != 'true' and parameter != 'False' and parameter != 'false':
                parameter = re.sub(ATOM_TYPE_REGEX, r'"\1"', parameter)

            parameter = parameter.replace('"none"','None')
            parameter = parameter.replace('"None"','None')
            parameter = parameter.replace('"True"','True')
            parameter = parameter.replace('"true"','True')
            parameter = parameter.replace('"False"','False')
            parameter = parameter.replace('"false"','False')
            parameter = parameter.replace('none','None')
            parameter = parameter.replace('false','False')
            parameter = parameter.replace('true','True')

            try:
                self.config[parameter_name] = ast.literal_eval(parameter)
            except Exception as e:
                raise configurationError(f"Error parsing '{parameter_name}': '{parameter}'")

    def checkInputParameters(self):

        for parameter_name,parameter in self.config.items():
            if parameter is None:
                continue
            if isinstance(self.types[parameter_name],list):
                matches = False
                for parameter_type in self.types[parameter_name]:
                    if isinstance(parameter,parameter_type):
                        matches = True
                        break
                if not matches:
                    raise configurationError(f"Invalid input for '{parameter_name}': {self.config[parameter_name]}, syntax is {self.syntax[parameter_name]}")
            elif not isinstance(parameter,self.types[parameter_name]):
                raise configurationError(f"Invalid input for '{parameter_name}': {self.config[parameter_name]}, syntax is {self.syntax[parameter_name]}")
    
    def writeFullInput(self):
        
        for parameter_name,parameter, in self.config.items():
            self.CONFIGOUT.write(f'{parameter_name} = {parameter}\t\t#{self.comments[parameter_name]}, syntax- {self.syntax[parameter_name]}\n')
        self.CONFIGOUT.flush()

    def outputType(self,graph_analysis=None):
        path = []
        if self.configExists('WRITE_DIRECTORY'):
            path.append(self.config['WRITE_DIRECTORY'])
        if self.configExists('RUN_NAME'):
            path.append(self.config['RUN_NAME'])
        path = os.path.join('',*path)
        if self.RANK == 0 and path:
            os.makedirs(path, exist_ok=True)
        self.COMM.Barrier()

        if "SLURM_JOB_ID" in os.environ and graph_analysis is None:
            if self.RANK == 0:
                self.STDOUT = open(os.path.join(path,'out'),'w')
                self.STDERR = open(os.path.join(path,'err'),'w')
                self.METAOUT = open(os.path.join(path,'meta'),'w')
                self.CONFIGOUT = open(os.path.join(path,'config_'),'w')
                self.STDOUT.write(f'SLURM JOD ID: {os.environ["SLURM_JOB_ID"]}')
                self.writeFullInput()
            self.COMM.Barrier()
            return

        self.STDOUT = sys.stdout
        self.STDERR = sys.stderr
        if graph_analysis is None:
            self.CONFIGOUT = open(os.path.join(path,'config_'),'w')

    def profile(self,on=0):

        if self.configExists('PROFILE') and self.config['PROFILE']:
            if on:
                self.profiler.enable()
                return
         
            self.profiler.disable()

            import io,pstats
            # Create a StringIO object to capture the output
            s = io.StringIO()

            # Create a Stats object and print the statistics
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            ps.print_stats()

            # Print the captured output
            if self.RANK == 0:
                self.STDOUT.write(s.getvalue(),flush=True)

def main():
    COMM = MPI.COMM_WORLD
    NP = COMM.Get_size()
    RANK = COMM.Get_rank()

    parser = argparse.ArgumentParser(prog='mySystem_mpi',description="Process user input file (-i INPUT_FILE_PATH) or make template input file (-m)")
    #group = parser.add_mutually_exclusive_group(required=True)
    group = parser.add_argument_group()

    # We map the input flag to the internal name 'file_path'
    group.add_argument(
        '-i', '--input', 
        type=str, 
        dest='input_file_path',
        help='Path to the input file to process'
    )

    group.add_argument(
        '-m', '--make_input', 
        action='store_true', 
        dest='generate_input',
        help='Generate a template input file'
    )

    group.add_argument(
        '-g', '--read_graph', 
        type=str, 
        dest='read_graph',
        help='Read a reaction graph file from a previous run'
    )

    args = parser.parse_args()

    if args.generate_input:
        config_manager = configuration()
        config_manager.writeFullInput()
        exit()
    
    if not args.input_file_path:
        exit()

    config_manager = configuration(args.input_file_path)

    #config_manager.loadCommandLine()
    config_manager.readInputFile()
    config_manager.checkInputParameters()
    config_manager.outputType(args.read_graph)
    config_manager.profile(1)
    if RANK == 0:
        if NP > 1:
            config_manager.STDOUT.write(f'Utilizing {NP} processes\n')
        else:
            config_manager.STDOUT.write(f'Utilizing {NP} process\n')

    if args.read_graph:
        if not os.path.isfile(args.read_graph):
            config_manager.STDOUT.write(f"File '{args.read_graph}' not found")
            exit()
        graph_read_system = system(config_manager)
        graph_read_system.readGraphFile(args.read_graph)
        graph_read_system.graphAnalysis()
        config_manager.profile(0)
        exit()

    #Create, initialize, and run Reaction System
    reaction_system = system(config_manager)

    #Initialize system from data file
    reaction_system.initialize()
    #Read/process system trajectory file
    reaction_system.run()

    #Wrap up processes
    reaction_system.finallize()
    config_manager.profile(0)

    exit()

if __name__ == "__main__":
    main()