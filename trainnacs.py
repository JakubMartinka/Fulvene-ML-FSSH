#!/usr/bin/env python3
'''
This code is used to phase-correct a dataset. It was modified many times during
implementation and has only been partially cleaned of redundant code.

Authors:
  Jakub Martinka based on scripts by Pavlo O. Dral and Lina Zhang, Fuchun Ge, Baoxin Xue
  20 May 2025
'''
import math, copy, random, os, rmsd, time
import numpy as np
import mlatom as ml

el2mass={
    "H": 1.007825,
    "C": 12.0
}

# ALIGN/ROTATE/CAP --------------------------------
def align_nacs(db, eq, nacs, rottype = 'kabsh', backalign = False):
    geoms = [mol.get_xyz_coordinates() for mol in db.molecules]
    geoms_aligned = []; nacs_aligned = []
    if rottype == 'toi': # tensor of inertia
        atom_labels = db[0].get_element_symbols()
        if not backalign:
            for igeom in range(len(nacs)):
                rotM, back_rotM = get_rot_matrix_ToI(geoms[igeom], atom_labels)
                nacs_aligned.append(rotate(nacs[igeom], rotM))
                geoms_aligned.append(rotate(geoms[igeom], rotM))
        else:
            for igeom in range(len(nacs)):
                rotM, back_rotM = get_rot_matrix_ToI(geoms[igeom], atom_labels)
                nacs_aligned.append(rotate(nacs[igeom], back_rotM))
                geoms_aligned.append(rotate(geoms[igeom], back_rotM))
    elif rottype == 'kabsh': # kabsh to equilibrium
        if not backalign:
            for igeom in range(len(nacs)):
                rotM, back_rotM = get_rot_matrix_kabsh(geoms[igeom], eq)
                nacs_aligned.append(rotate(nacs[igeom], rotM))
                geoms_aligned.append(rotate(geoms[igeom], rotM))
        else:
            for igeom in range(len(nacs)):
                rotM, back_rotM = get_rot_matrix_kabsh(geoms[igeom], eq)
                nacs_aligned.append(rotate(nacs[igeom], back_rotM))
                geoms_aligned.append(rotate(geoms[igeom], back_rotM))

    return np.array(nacs_aligned), np.array(geoms_aligned)

def get_rot_matrix_kabsh(geom, refgeom):
    mol = np.array(geom)
    ref = np.array(refgeom)
    mol = mol - getCoM(mol)
    ref = ref - getCoM(ref)
    rotM = rmsd.kabsch(mol, ref) 
    back_rotM = rmsd.kabsch(ref, mol)
    return rotM, back_rotM

def get_rot_matrix_ToI(geom, atom_labels):
    mol = np.array(geom)
    masses = np.array([el2mass[label] for label in atom_labels])
    CoM = getCoM(mol, masses)
    mol = mol - CoM
    ToI = getToI(mol, masses)
    principal_moments, rotM = np.linalg.eigh(ToI)
    return rotM, rotM.T

def getToI(xyz, m):
    inertia_tensor = np.zeros((3, 3))
    for i in range(xyz.shape[0]):
        x, y, z = xyz[i]
        inertia_tensor[0, 0] += m[i] * (y**2 + z**2)  # Ixx
        inertia_tensor[1, 1] += m[i] * (x**2 + z**2)  # Iyy
        inertia_tensor[2, 2] += m[i] * (x**2 + y**2)  # Izz
        inertia_tensor[0, 1] += -m[i] * x * y  # Ixy and Iyx
        inertia_tensor[0, 2] += -m[i] * x * z  # Ixz and Izx
        inertia_tensor[1, 2] += -m[i] * y * z  # Iyz and Izy
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]
    return inertia_tensor

def getCoM(xyz, m=None):
    if m is None:
        m = np.ones(xyz.shape[0])
    return np.sum(xyz*m[:,np.newaxis],axis=0)/np.sum(m)

def rotate(coords, rotM):
    return np.array(coords).dot(rotM)

def cap_nacs(db, nacs, Hstate, Lstate):
    enH = [mol.electronic_states[Hstate].energy for mol in db.molecules]
    enL = [mol.electronic_states[Lstate].energy for mol in db.molecules]
    dE = [enH[ii] - enL[ii] for ii in range(len(enH))]
    for igeom in range(len(nacs)):
        for iatom in range(len(nacs[igeom])):
            for icoord in range(len(nacs[igeom][iatom])):
                nacs[igeom][iatom][icoord] *= dE[igeom]
    return nacs

def uncap_nacs(db, nacs, Hstate, Lstate):
    enH = [mol.electronic_states[Hstate].energy for mol in db.molecules]
    enL = [mol.electronic_states[Lstate].energy for mol in db.molecules]
    dE = [enH[ii] - enL[ii] for ii in range(len(enH))]
    for igeom in range(len(nacs)):
        for iatom in range(len(nacs[igeom])):
            for icoord in range(len(nacs[igeom][iatom])):
                nacs[igeom][iatom][icoord] /= dE[igeom]
    return nacs
# -------------------------------------------------

# READ/WRITE --------------------------------------
def save_nacs_db(db, nacs, db_name_pc, Hstate, Lstate):
    for imol, mol in enumerate(db.molecules):
        for iat, atom in enumerate(mol.atoms):
            atom.nonadiabatic_coupling_vectors[Lstate][Hstate] = np.array(nacs[imol][iat])
            atom.nonadiabatic_coupling_vectors[Hstate][Lstate] = -np.array(nacs[imol][iat])
    db.dump(db_name_pc, format="json")
# -------------------------------------------------

# DESCRIPTORS -------------------------------------
def get_descriptors(desc_params):
    db = desc_params['db']
    Xs = []
    if 'RE' in desc_params['descriptor']:
        re = [getRE(mol, desc_params['eq_db'][0]) for mol in db.molecules]
        Xs.append(re)
    if 'dE' in desc_params['descriptor']:
        enH = [mol.electronic_states[desc_params['Hstate']].energy for mol in db.molecules]
        enL = [mol.electronic_states[desc_params['Lstate']].energy for mol in db.molecules]
        dE = [enH[ii] - enL[ii] for ii in range(len(enH))]
        Xs.append(dE)
    dgrad = [mol.electronic_states[desc_params['Hstate']].energy_gradients - mol.electronic_states[desc_params['Lstate']].energy_gradients for mol in db.molecules]
    rot_dgrad, _ = align_nacs(db, desc_params['eq_db'][0].get_xyz_coordinates(), dgrad, desc_params['rottype'])
    if 'ddgrad' in desc_params['descriptor']:
        Xs.append([idgrad.reshape(-1).tolist() for idgrad in rot_dgrad])
    if 'adgrad' in desc_params['descriptor']:
        Xs.append([np.abs(idgrad).reshape(-1).tolist() for idgrad in rot_dgrad])
    if 'ndgrad' in desc_params['descriptor']:
        Xs.append([np.linalg.norm(idgrad, 'fro') for idgrad in rot_dgrad])
    Xs = [[[item] if not isinstance(item, list) else item for item in lst] for lst in Xs]
    Xs = [[item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])] for lst in [sum(items, []) for items in zip(*Xs)]]

    if desc_params['save_descriptors']:
        with open(f'{os.getcwd()}/train.x_{desc_params["descriptor"]}', 'w') as f:
            for iX in Xs:
                descstr = ''
                for i in iX:
                    descstr += f'   {i:.13f}'
                f.writelines(f'{descstr.strip()}\n')
    return Xs

def getRE(mol, eq_mol):
    RE = []
    for iatom in range(len(mol.atoms)):
        for jatom in range(iatom+1, len(mol.atoms)):
            ij_R = np.sqrt((mol.atoms[iatom].xyz_coordinates[0] - mol.atoms[jatom].xyz_coordinates[0])**2 + (mol.atoms[iatom].xyz_coordinates[1] - mol.atoms[jatom].xyz_coordinates[1])**2 + (mol.atoms[iatom].xyz_coordinates[2] - mol.atoms[jatom].xyz_coordinates[2])**2)
            ij_Req = np.sqrt((eq_mol.atoms[iatom].xyz_coordinates[0] - eq_mol.atoms[jatom].xyz_coordinates[0])**2 + (eq_mol.atoms[iatom].xyz_coordinates[1] - eq_mol.atoms[jatom].xyz_coordinates[1])**2 + (eq_mol.atoms[iatom].xyz_coordinates[2] - eq_mol.atoms[jatom].xyz_coordinates[2])**2)
            RE.append(ij_Req/ij_R)
    return RE
# -------------------------------------------------

# PHASE CORRECTION --------------------------------
def correct_phase(train_params):
    NACs = [[atom.nonadiabatic_coupling_vectors[train_params['Lstate']][train_params['Hstate']] for atom in mol.atoms] for mol in train_params['db'].molecules]

    print("time_5 = ", time.time(), "# before nacs alignment")
    NACs_aligned, _ = align_nacs(train_params['db'], train_params['eq_db'][0].get_xyz_coordinates(), NACs, train_params['rottype'])
    print("time_6 = ", time.time(), "# after nacs alignment, before cap")
    NACs_aligned_capped = cap_nacs(train_params['db'], NACs_aligned, train_params['Hstate'], train_params['Lstate'])
    print("time_7 = ", time.time(), "# after nacs cap, before get abs")
    NACs_aligned_capped_abs = get_nacs_abs(NACs_aligned_capped)
    print("time_8 = ", time.time(), "# after get abs, before opt hyperparameters for abs nac")
    lamda, sigma = opt_hyperparams(NACs_aligned_capped_abs, train_params['X'])
    print("time_9 = ", time.time(), "# after opt hyperparameters for abs nac")
    print("  Lambda: ", lamda, " Sigma: ", sigma)

    pseudo_nacs = copy.deepcopy(NACs_aligned_capped)
    
    nothing_to_flip = False; iouterloop = 0
    if train_params['maxNiter'] == 0:
        nothing_to_flip = True

    test_value = []
    print("time_10 = ", time.time(), "# before phase-correction")
    while not nothing_to_flip:
        iall = [ii for ii in range(len(NACs_aligned_capped_abs))]
        random.shuffle(iall)
        
        indices = np.arange(len(train_params["db"]))

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        nothing_to_flip = True
        for ii, (train_index, test_index) in enumerate(kf.split(indices), 1):
            os.system('rm -f nacs.npz')
            
            print(f"Time_11_{iouterloop}_{ii} = ", time.time(), "# before train on one fold, before one iter of PC")
            mlDB = ml.data.ml_database()
            mlDB.x = np.array(train_params['X'])
            mlDB.y = np.array([np.reshape(nac, nac.shape[0]*3) for nac in pseudo_nacs])
            
            model = ml.models.krr(model_file="nacs.npz", kernel_function='Gaussian', ml_program='mlatom.jl')
            model.hyperparameters['sigma'].value = sigma
            model.hyperparameters['lambda'].value = lamda
            
            train, test = mlDB.split(number_of_splits=2, sampling='user-defined', indices=[list(train_index), list(test_index)])
            model.train(ml_database=train)
            print(f"Time_11_{iouterloop}_{ii} = ", time.time(), "# after train on one fold")
            model.predict(ml_database=mlDB,property_to_predict='nacs')
            model.predict(ml_database=test,property_to_predict='nacs')

            y = test.y
            yest = test.get_property('nacs')
            RMSE = ml.stats.rmse(yest.reshape(yest.size),y.reshape(y.size))
            R2 = ml.stats.correlation_coefficient(yest.reshape(yest.size),y.reshape(y.size))**2
            
            pseudo_nacs_est = [np.reshape(nac, (int(nac.shape[0]/3), 3)) for nac in mlDB.get_property('nacs')]
            n_flipped = 0

            ref = copy.deepcopy(pseudo_nacs)
            est = copy.deepcopy(pseudo_nacs_est)
            ref = uncap_nacs(train_params['db'], ref, train_params['Hstate'], train_params['Lstate'])
            est = uncap_nacs(train_params['db'], est, train_params['Hstate'], train_params['Lstate'])

            for igeom in range(len(pseudo_nacs)):
                dotprod = 0; sm = 0; mse1 = 0; mse2 = 0
                for iatom in range(len(pseudo_nacs[igeom])):
                    for icoord in range(len(pseudo_nacs[igeom][iatom])):
                        dotprod += ref[igeom][iatom][icoord] * est[igeom][iatom][icoord]
                        sm += np.sign(ref[igeom][iatom][icoord]) * np.sign(est[igeom][iatom][icoord])
                        mse1 += ( ref[igeom][iatom][icoord] - est[igeom][iatom][icoord])**2
                        mse2 += (-ref[igeom][iatom][icoord] - est[igeom][iatom][icoord])**2
                if mse2 < mse1:
                        print(f'  |- flip {igeom}, dotprod={dotprod:.13f}, mse1={mse1:.13f}, mse2={mse2:.13f}')
                        n_flipped += 1
                        for iatom in range(len(pseudo_nacs[igeom])):
                            for icoord in range(len(pseudo_nacs[igeom][iatom])):
                                pseudo_nacs[igeom][iatom][icoord] = -1 * pseudo_nacs[igeom][iatom][icoord]
            print(f'  iteration {iouterloop+1}, flipped {n_flipped} NACs, R^2 = {R2}, RMSE = {RMSE}')
            print(' ' + '-' * 82)
            print(f"Time_12_{iouterloop}_{ii} = ", time.time(), "# after one iter of phase-correction")

            if n_flipped != 0:
                nothing_to_flip = False

        if n_flipped == 1:
            test_value.append(igeom)
            if len(test_value) > 1 and test_value[-1] == test_value[-2]:
                print("time_13 = ", time.time(), "# patience threshold")
                nothing_to_flip = True

        iouterloop += 1
        os.system('rm -f nacs.npz')
        if iouterloop == maxNiter:
            print("time_13 = ", time.time(), "# max iteration reached")
            break
        if nothing_to_flip:
            print("time_13 = ", time.time(), "# nothing to flip")
            print("Nothing more to flip! :-(")
    
    print("time_14 = ", time.time(), "# before fitting final model")
    mlDB = ml.data.ml_database()
    mlDB.x = np.array(train_params['X'])
    mlDB.y = np.array([np.reshape(nac, np.array(nac).shape[0]*3) for nac in pseudo_nacs])
    
    model = ml.models.krr(model_file=f'fin_{train_params["setSize"]}_{train_params["descriptor"]}_nacs.npz', kernel_function='Gaussian', ml_program='mlatom.jl')

    model.hyperparameters['sigma'].minval = 2**-5
    model.hyperparameters['sigma'].maxval = 2**9
    model.hyperparameters['sigma'].optimization_space = 'log'  
    model.hyperparameters['lambda'].minval = 2**-35
    model.hyperparameters['lambda'].maxval = 1.0
    model.hyperparameters['lambda'].optimization_space = 'log'
    sub, val = mlDB.split(number_of_splits=2, fraction_of_points_in_splits=[0.8, 0.2], sampling='random')
    model.optimize_hyperparameters(
        subtraining_ml_database=sub,
        validation_ml_database=val,
        optimization_algorithm='grid',
        hyperparameters=['lambda','sigma'],
        debug=False
        )
    model.train(ml_database=mlDB)
    print("time_15 = ", time.time(), "# after fitting final model")
    model.predict(ml_database=mlDB,property_to_predict='nacs_pred')
    y = mlDB.y
    yest = mlDB.get_property('nacs_pred')
    print('---------------------------------------------------------------')
    print('Predictions for training set! This is NOT an independent TEST SET!')
    print('ref:', np.array(y[0]).flatten())
    print('pred:', np.array(yest[0]).flatten())
    RMSE = ml.stats.rmse(yest.reshape(yest.size),y.reshape(y.size))
    R2 = ml.stats.correlation_coefficient(yest.reshape(yest.size),y.reshape(y.size))**2
    print("RMSE: ", RMSE, ",R2: ", R2)

    NACs_aligned_capped_pc = copy.deepcopy(pseudo_nacs)
    print("time_16 = ", time.time(), "# before uncap")
    NACs_aligned_pc = uncap_nacs(train_params["db"], NACs_aligned_capped_pc, train_params["Hstate"], train_params["Lstate"])
    print("time_17 = ", time.time(), "# after uncap, before back rotate")
    NACs_pc, _ = align_nacs(train_params["db"], train_params["eq_db"][0].get_xyz_coordinates(), NACs_aligned_pc, train_params["rottype"], backalign = True)
    print("time_18 = ", time.time(), "# after back rotate")

    return NACs_pc

def get_nacs_abs(nacs):
    nacs_abs = copy.deepcopy(nacs)
    for igeom in range(len(nacs_abs)):
        for iatom in range(len(nacs_abs[igeom])):
            for icoord in range(len(nacs_abs[igeom][iatom])):
                nacs_abs[igeom][iatom][icoord] = abs(nacs_abs[igeom][iatom][icoord])
    return nacs_abs

def opt_hyperparams(nacs_abs, X):
    mlDB = ml.data.ml_database()
    mlDB.x = np.array(X)
    mlDB.y = np.array([np.reshape(nac, nac.shape[0]*3) for nac in nacs_abs])
    model = ml.models.krr(model_file="nacs_abs.npz", kernel_function='Gaussian', ml_program='mlatom.jl')
    model.hyperparameters['sigma'].minval = 2**-5
    model.hyperparameters['sigma'].maxval = 2**9
    model.hyperparameters['sigma'].optimization_space = 'log'  
    model.hyperparameters['lambda'].minval = 2**-35
    model.hyperparameters['lambda'].maxval = 1.0
    model.hyperparameters['lambda'].optimization_space = 'log'
    sub, val = mlDB.split(number_of_splits=2, fraction_of_points_in_splits=[0.8, 0.2], sampling='random')
    model.optimize_hyperparameters(
        subtraining_ml_database=sub,
        validation_ml_database=val,
        optimization_algorithm='grid',
        hyperparameters=['lambda','sigma'],
        debug=False
        )
    return model.hyperparameters['lambda'].value, model.hyperparameters['sigma'].value

def create_model(train_params, model_name="nacs.npz"):
    NACs = [[atom.nonadiabatic_coupling_vectors[train_params['Lstate']][train_params['Hstate']] for atom in mol.atoms] for mol in train_params['db'].molecules]
    NACs_aligned, _ = align_nacs(train_params['db'], train_params['eq_db'][0].get_xyz_coordinates(), NACs, train_params['rottype'])
    NACs_aligned_capped = cap_nacs(train_params['db'], NACs_aligned, train_params['Hstate'], train_params['Lstate'])

    mlDB = ml.data.ml_database()
    mlDB.x = np.array(X)
    mlDB.y = np.array([np.reshape(nac, np.array(nac).shape[0]*3) for nac in NACs_aligned_capped])
    
    model = ml.models.krr(model_file=model_name, kernel_function='Gaussian', ml_program='mlatom.jl')

    model.hyperparameters['sigma'].minval = 2**-5
    model.hyperparameters['sigma'].maxval = 2**9
    model.hyperparameters['sigma'].optimization_space = 'log'  
    model.hyperparameters['lambda'].minval = 2**-35
    model.hyperparameters['lambda'].maxval = 1.0
    model.hyperparameters['lambda'].optimization_space = 'log'
    sub, val = mlDB.split(number_of_splits=2, fraction_of_points_in_splits=[0.8, 0.2], sampling='random')
    model.optimize_hyperparameters(
        subtraining_ml_database=sub,
        validation_ml_database=val,
        optimization_algorithm='grid',
        hyperparameters=['lambda','sigma'],
        debug=False
        )
    model.train(ml_database=mlDB)
    model.predict(ml_database=mlDB,property_to_predict='nacs_pred')
    y = mlDB.y
    yest = mlDB.get_property('nacs_pred')
    print('---------------------------------------------------------------')
    print('Predictions for training set! This is NOT an independent TEST SET!')
    print('ref:', np.array(y[0]).flatten())
    print('pred:', np.array(yest[0]).flatten())
    RMSE = ml.stats.rmse(yest.reshape(yest.size),y.reshape(y.size))
    R2 = ml.stats.correlation_coefficient(yest.reshape(yest.size),y.reshape(y.size))**2
    print("RMSE: ", RMSE, ",R2: ", R2)
# -------------------------------------------------

if __name__ == '__main__':
    
    # Relative to Equilibrium (RE) + gradient difference (ddgrad)
    descriptor = 'RE_ddgrad'
    
    #db_name = # name of database to be phase-corrected
    db_name = '100_test.json'
    eq_name = 'eqmol_fulvene.json'

    path_to_set = os.getcwd() + '/'
    
    print("time_1 = ", time.time(), "# start")
    db = ml.data.molecular_database.load(filename=path_to_set + db_name, format='json')
    print("time_2 = ", time.time(), "# after loading database")
    eq_db = ml.data.molecular_database.load(filename=path_to_set + eq_name, format='json')

    # In case of Fulvene, only two states necessary
    Hstate = 1 # second excited state
    Lstate = 0 # first excited state
    maxNiter = 100 # maxNiter = 0 if only training is needed
    rottype = 'kabsh' # type of reference frame rotation

    desc_params = {
        'db': db,
        'eq_db': eq_db,
        'descriptor': descriptor,
        'Hstate': Hstate,
        'Lstate': Lstate,
        'rottype': rottype,
        'save_descriptors': False
    }
    print("time_3 = ", time.time(), "# before making descriptors")
    X = get_descriptors(desc_params)
    print("time_4 = ", time.time(), "# after making descriptors")

    train_params = {
        'db': db,
        'eq_db': eq_db,
        'maxNiter': maxNiter,
        'setSize': len(db),
        'descriptor': descriptor,
        'X': X,
        'Hstate': Hstate,
        'Lstate': Lstate,
        'rottype': rottype,
        'phase_correct': False
    }
    
    # Phase correction
    nacs_pc = correct_phase(train_params)

    # Save phase corrected database
    db_name_pc = f'pc_{db_name}'
    save_nacs_db(db, nacs_pc, db_name_pc, Hstate, Lstate)
    
    # Training model on database
    #create_model(train_params, f"fin_{descriptor}_nacs.npz")
