'''
CRO for LennardJones Clusters
by Martin Moreno and Luis Rincon
Last updated: August 25th, 2022

Optimization for Lenard Jones Clusters
using directed montecarlo and BFLGS
'''
#Import libraries
import numpy as np
np.random.seed()
from scipy.optimize import minimize
from numba import njit
import time
import matplotlib.pyplot as plt



@njit
def ene_lj(w):
    """The Lennard-Jones Potential"""
    natoms = int(len(w)/3)                                        # number of atoms
    ene = 0.0
    for ix in range(natoms-1):
        for jx in range(ix+1,natoms):                                # loop over atoms pairs
            xr = w[3*ix] - w[3*jx]
            yr = w[3*ix+1] - w[3*jx+1]
            zr = w[3*ix+2] - w[3*jx+2]
            r2 = np.power(xr,2) + np.power(yr,2) + np.power(zr,2)
            if r2<=0.010:
                r2 = 0.010
            r2i = 1.0/r2
            r6i = np.power(r2i,3)
            ene = ene + 4.0*r6i*(r6i-1.0)  # update energy
    return ene + offset

def dene_lj(w):
    """The Lennard-Jones gradient vector"""
    natoms = int(len(w)/3)                                         # number of atoms
    dene = np.zeros_like(w)                                    # set gradient to zero
    for ix in range(natoms-1):
        for jx in range(ix+1,natoms):                                # loop over atoms pairs
            xr = w[3*ix] - w[3*jx]
            yr = w[3*ix+1] - w[3*jx+1]
            zr = w[3*ix+2] - w[3*jx+2]
            r2 = np.power(xr,2) + np.power(yr,2) + np.power(zr,2)
            if r2==0.0:
                r2=0.10
            r2i = 1.0/r2
            r6i = np.power(r2i,3)
            df = -48.0*r2i*r6i*(r6i-0.5)
            dene[3*ix] = dene[3*ix] + df*xr                     # update gradients
            dene[3*ix+1] = dene[3*ix+1] + df*yr
            dene[3*ix+2] = dene[3*ix+2] + df*zr
            dene[3*jx] = dene[3*jx] - df*xr
            dene[3*jx+1] = dene[3*jx+1] - df*yr
            dene[3*jx+2] = dene[3*jx+2] - df*zr
    return dene

@njit
def ene_lj_atoms(w):
    """Lenard Jones energy for each atoms"""
    natoms = int(len(w)/3)
    atoms_energy = []
    for index1 in range(natoms): #loop for all atoms
        ene = 0.0
        for index2 in range(natoms): #loop for all atoms
            if index1!=index2:
                xr = w[3*index1] - w[3*index2]
                yr = w[3*index1+1] - w[3*index2+1]
                zr = w[3*index1+2] - w[3*index2+2]
                r2 = np.power(xr,2) + np.power(yr,2) + np.power(zr,2)
                if r2<=0.010:
                    r2 = 0.010
                r2i = 1.0/r2
                r6i = np.power(r2i,3)
                ene = ene + 2.0*r6i*(r6i-1.0)
        atoms_energy.append(ene)
    atoms_energy = np.array(atoms_energy)
    return atoms_energy  #return an array with the energy for each atom
@njit
def ene_lj_atom(w,xn,yn,zn,o):
    """ Calculate atom indiviudal energy based on xn,yn,zn positions"""
    natoms = int(len(w)/3)
    ene = 0.0
    for index2 in range(natoms):
        if index2!=o:
            xr = xn - w[3*index2]
            yr = yn - w[3*index2+1]
            zr = zn - w[3*index2+2]
            r2 = np.power(xr,2) + np.power(yr,2) + np.power(zr,2)
            if r2<=0.010:
                r2 = 0.010
            r2i = 1.0/r2
            r6i = np.power(r2i,3)
            ene = ene + 2.0*r6i*(r6i-1.0)
    return ene #returns the single atom "o" energy

def grid(m1,q):
    '''Obtain an array with the vertices of the target atom'''
    atomx = m1[3*q]
    atomy = m1[3*q+1]
    atomz = m1[3*q+2]
    atom = np.array([atomx,atomy,atomz])
    listbi = []
    vertices = np.zeros(24)
    for vertex in range(8):
        for bit in range(3):
            bi = (vertex>>bit)&1
            listbi.append(bi)
    listbi = np.array(listbi)
    for vertex in range(8):
        vertices[3*vertex] = atom[0] + ((-1)**listbi[3*vertex])*r_grid_atom
        vertices[3*vertex+1] = atom[1] + ((-1)**listbi[3*vertex+1])*r_grid_atom
        vertices[3*vertex+2] = atom[2] + ((-1)**listbi[3*vertex+2])*r_grid_atom
    vertices = np.array(vertices)
    return vertices

def cm(x):
    """center of mass"""
    natoms = int(len(x)/3)
    xcm = 0.0
    ycm = 0.0
    zcm = 0.0
    for i in range(natoms):
        xcm = xcm + x[3*i]
        ycm = ycm + x[3*i+1]
        zcm = zcm + x[3*i+2]
    xcm = xcm/natoms
    ycm = ycm/natoms
    zcm = zcm/natoms
    return xcm,ycm,zcm

#Unimolecular operators
def operator_decomposition(m1,i):
    #molecule 1 is the same molecule
    x0 = np.copy(m1)
    opt = minimize(ene_lj,x0, method='BFGS', jac=dene_lj) #x0 is the initial guess obtained from CRO
    m2 = opt.x #optimize and store the result in m2
    xcm,ycm,zcm = cm(m2) #calculate the mass center
    for icm in range(atoms):
        m2[3*icm] = m2[3*icm] - xcm
        m2[3*icm+1] = m2[3*icm+1] - ycm
        m2[3*icm+2] = m2[3*icm+2] - zcm
    ene1 = PotentialEnergy[i]  #calculate the energy for molecule 1 no opt
    ene2 = opt.fun #calculate the energy for molecule 2
    if ene2 > ene1:
        m2 = np.copy(m1)
        ene2 = ene1
    return  m1,m2,ene1,ene2 #return the evaluated energies for avoiding reducing comp time

def operator_unicollision(m1):
    #Directed Montecarlo
    oli = 0
    for it in range(n_mc): #Montecarlo simulation for molecule 1
        atoms_ene = ene_lj_atoms(m1) #calculate the energy for each atom
        weight_active = np.exp(atoms_ene/Eactive) #calculate the weight active
        weight_target = np.exp(-((atoms_ene-Etarget)**2)/(2*SigmaTarget**2)) #calculate the weight target
        probability_active = weight_active/np.sum(weight_active) #calculate the probability active
        probability_target = weight_target/np.sum(weight_target) #calculate the probability target
        o = np.random.choice(index,p = probability_active) #select a random atom based on its energy
        ene_active = atoms_ene[o] #store the energy value for the atom
        q = np.random.choice(index,p = probability_target) #select a random atom as target
        if o != q:
            vert = grid(m1,q) #obtain the vertices from a grid for the target atom
            vertEnergies = np.zeros(8)
            for numVert in range(8):
                xn = vert[3*numVert]
                yn = vert[3*numVert+1]
                zn = vert[3*numVert+2]
                vertEnergies[numVert] = ene_lj_atom(m1,xn,yn,zn,o)
            ene_active_n = vertEnergies.min()
            bestVert = np.where(vertEnergies == ene_active_n)
            bestVert = bestVert[0]
            xn = vert[3*bestVert]
            yn = vert[3*bestVert+1]
            zn = vert[3*bestVert+2]
            dene = ene_active_n - ene_active #calculate the delta energy
            if dene < 0: #update the cluster if delta energy es smaller than 0
                m1[3*o] = xn
                m1[3*o+1] = yn
                m1[3*o+2] = zn
                oli = oli + 1
            else: #if the delta energy is greater than 0
                shi = np.random.rand() #random number for comparing with probabiliy
                prob = np.exp(-dene/EaDMC) #probabiliy for considering the move
                if shi < prob: #update the cluster if shi is smaller than the probability
                    m1[3*o] = xn
                    m1[3*o+1] = yn
                    m1[3*o+2] = zn
                    oli = oli + 1
    #Calculate the energy for the new structure
    ene = ene_lj(m1)
    return m1,ene


#Intermolecular operators

def operator_synthesis(m1,m2,i,j):
    ene1_0 = PotentialEnergy[i] #store the energy for the molecule 1 before being optimized
    m1_0 = np.copy(m1) #obtain a copy of the structure before optimizing molecule 1
    opt = minimize(ene_lj,m1_0, method='BFGS', jac=dene_lj) #x0 is the initial guess obtained from CRO
    m1 = opt.x #optimize and store the result in m1
    xcm,ycm,zcm = cm(m1) #calculate the mass center
    for icm in range(atoms):
        m1[3*icm] = m1[3*icm] - xcm
        m1[3*icm+1] = m1[3*icm+1] - ycm
        m1[3*icm+2] = m1[3*icm+2] - zcm
    ene1 = opt.fun #obtain the optimized value for the molecule
    ene2_0 = PotentialEnergy[j] #store the energy for the molecule 2 before
    #optimization
    m2_0 = np.copy(m2) #obtain a copy of the structure before optimizing molecule 2
    opt = minimize(ene_lj,m2_0, method='BFGS', jac=dene_lj) #x0 is the initial guess obtained from CRO
    m2 = opt.x #optimize and store the result in m2
    xcm,ycm,zcm = cm(m2) #calculate the mass center
    for icm in range(atoms):
        m2[3*icm] = m2[3*icm] - xcm
        m2[3*icm+1] = m2[3*icm+1] - ycm
        m2[3*icm+2] = m2[3*icm+2] - zcm
    ene2 = opt.fun #obtain optimized value for the molecule
    mprime = 0 #initialize the variable for entering in the if statemente
    eneprime = 0 #initialize the variable for entering in the if statemente
    #Conditions for preventing the local optimziation algorithm to break
    if ene1 > ene1_0:
        m1 = m1_0
        ene1 = ene1_0
    if ene2 > ene2_0:
        m2 = m2_0
        ene2 = ene2_0
    if ene1 < ene2:
        mprime = m1
        eneprime = ene1
    else:
        mprime = m2
        eneprime = ene2
    return mprime,eneprime

def operator_collision(m1,m2):
    #Directed Montecarlo
    oli1 = 0
    for it in range(n_mc): #Montecarlo simulation for molecule 1
        atoms_ene = ene_lj_atoms(m1) #calculate the energy for each atom
        weight_active = np.exp(atoms_ene/Eactive) #calculate the weight active
        weight_target = np.exp(-((atoms_ene-Etarget)**2)/(2*SigmaTarget**2)) #calculate the weight target
        probability_active = weight_active/np.sum(weight_active) #calculate the probability active
        probability_target = weight_target/np.sum(weight_target) #calculate the probability target
        o = np.random.choice(index,p = probability_active) #select a random atom based on its energy
        ene_active = atoms_ene[o] #store the energy value for the atom
        q = np.random.choice(index,p = probability_target) #select a random atom as target
        if o != q:
            vert = grid(m1,q) #obtain the vertices from a grid for the target atom
            vertEnergies = np.zeros(8)
            for numVert in range(8):
                xn = vert[3*numVert]
                yn = vert[3*numVert+1]
                zn = vert[3*numVert+2]
                vertEnergies[numVert] = ene_lj_atom(m1,xn,yn,zn,o)
            ene_active_n = vertEnergies.min()
            bestVert = np.where(vertEnergies == ene_active_n)
            bestVert = bestVert[0]
            xn = vert[3*bestVert]
            yn = vert[3*bestVert+1]
            zn = vert[3*bestVert+2]
            dene = ene_active_n - ene_active #calculate the delta energy
            if dene < 0: #update the cluster if delta energy es smaller than 0
                m1[3*o] = xn
                m1[3*o+1] = yn
                m1[3*o+2] = zn
                oli1 = oli1+1
            else: #if the delta energy is greater than 0
                shi = np.random.rand() #random number for comparing with probabiliy
                prob = np.exp(-dene/EaDMC) #probabiliy for considering the move
                if shi < prob: #update the cluster if shi is smaller than the probability
                    m1[3*o] = xn
                    m1[3*o+1] = yn
                    m1[3*o+2] = zn
                    oli1 = oli1+1

    #Calculate the energy for the new structure
    ene1 = ene_lj(m1)
    #Directed Montecarlo
    oli2 = 0
    for it in range(n_mc): #Montecarlo simulation for molecule 1
        atoms_ene = ene_lj_atoms(m2) #calculate the energy for each atom
        weight_active = np.exp(atoms_ene/Eactive) #calculate the weight active
        weight_target = np.exp(-((atoms_ene-Etarget)**2)/(2*SigmaTarget**2)) #calculate the weight target
        probability_active = weight_active/np.sum(weight_active) #calculate the probability active
        probability_target = weight_target/np.sum(weight_target) #calculate the probability target
        o = np.random.choice(index,p = probability_active) #select a random atom based on its energy
        ene_active = atoms_ene[o] #store the energy value for the atom
        q = np.random.choice(index,p = probability_target) #select a random atom as target
        if o != q:
            vert = grid(m2,q) #obtain the vertices from a grid for the target atom
            vertEnergies = np.zeros(8)
            for numVert in range(8):
                xn = vert[3*numVert]
                yn = vert[3*numVert+1]
                zn = vert[3*numVert+2]
                vertEnergies[numVert] = ene_lj_atom(m2,xn,yn,zn,o)
            ene_active_n = vertEnergies.min()
            bestVert = np.where(vertEnergies == ene_active_n)
            bestVert = bestVert[0]
            xn = vert[3*bestVert]
            yn = vert[3*bestVert+1]
            zn = vert[3*bestVert+2]
            dene = ene_active_n - ene_active #calculate the delta energy
            if dene < 0: #update the cluster if delta energy es smaller than 0
                m2[3*o] = xn
                m2[3*o+1] = yn
                m2[3*o+2] = zn
                oli2 = oli2 + 1
            else: #if the delta energy is greater than 0
                shi = np.random.rand() #random number for comparing with probabiliy
                prob = np.exp(-dene/EaDMC) #probabiliy for considering the move
                if shi < prob: #update the cluster if shi is smaller than the probability
                    m2[3*o] = xn
                    m2[3*o+1] = yn
                    m2[3*o+2] = zn
                    oli2 = oli2 + 1

    #Calculate the energy for the new structure
    ene2 = ene_lj(m2)
    return m1,m2,ene1,ene2

EaDMC_axis = np.arange(0.25,0.40,0.01)
MinPE_axis = np.zeros(EaDMC_axis.size)
Iterations_axis = np.zeros(EaDMC_axis.size)
ReportedValue = -173.9284266

for indexAxis in range(EaDMC_axis.size):
    st = time.time()
    #Set the parameters
    N = 10000 #iteration number
    PopSize = 20#population size, must be initialized as an integer
    KineticEnergyLossRate = 0.1 #kinetic energy reduction
    MoleColl = 0.2 #intermolecular probability
    buffer = 1.0E04 #surroundings energy
    atoms = 38  #number of atoms for the cluster
    alfa = 10
    beta = 10
    Eactive = 1.0 #parameter for directed montecarlo
    Etarget = -4.0 #parameter for directed montecarlo
    SigmaTarget = 1.5 #parameter for directed montecarlo
    index = np.arange(atoms) #indexes for the directed montecarlo
    r_grid = 0.6*np.power(float(atoms),1.0/3.0) #radius for an spherical grid
    r_grid_atom = 1.0/np.sqrt(3.0) #half side of the grid for directed montecarlo
    n_mc = atoms #iterations for the directed montecarlo
    offset = 173.93
    EaDMC = EaDMC_axis[indexAxis]
    
    #Initialization
    w = np.zeros((PopSize,3*atoms))
    PotentialEnergy = np.zeros(PopSize)
    NumHit = np.zeros(PopSize,dtype = int)
    KineticEnergy = np.zeros(PopSize)
    ActualMinStruct = np.array(3*atoms)
    ActualMinPotentialEnergy = offset
    for i in range(PopSize):
        w[i,0] = r_grid*(2.0*np.random.rand()-1.0)
        w[i,1] = r_grid*(2.0*np.random.rand()-1.0)
        w[i,2] = r_grid*(2.0*np.random.rand()-1.0)
        for j in range(1,atoms):
            check = True
            while check:
                check = False
                w[i,3*j] = r_grid*(2.0*np.random.rand()-1.0)
                w[i,3*j+1] = r_grid*(2.0*np.random.rand()-1.0)
                w[i,3*j+2] = r_grid*(2.0*np.random.rand()-1.0)
                for k in range(j):
                    dx = w[i,3*j]-w[i,3*k]
                    dy = w[i,3*j+1]-w[i,3*k+1]
                    dz = w[i,3*j+2]-w[i,3*k+2]
                    rij = np.sqrt(np.power(dx,2)+np.power(dy,2)+np.power(dz,2))
                    if rij<=1.0:
                        check=True
        opt = minimize(ene_lj,w[i,:], method='BFGS', jac=dene_lj) #x0 is the initial guess obtained from CRO
        w[i,:] = opt.x
        PotentialEnergy[i] = opt.fun
        KineticEnergy[i] = 1.0E05
        print(i,PotentialEnergy[i]-offset,KineticEnergy[i])
        xcm,ycm,zcm = cm(w[i,:])
        for j in range(atoms):
            w[i,3*j] = w[i,3*j] - xcm
            w[i,3*j+1] = w[i,3*j+1] - ycm
            w[i,3*j+2] = w[i,3*j+2] - zcm
        if PotentialEnergy[i] < ActualMinPotentialEnergy:
            ActualMinPotentialEnergy = PotentialEnergy[i]
            ActualMinStruct = w[i,:]

    MinStruct = np.copy(w) #create the MinStruct array
    MinHit = np.zeros(PopSize,dtype =int) #create the min hit array
    MinPotentialEnergy = np.copy(PotentialEnergy)
    print(f'ActualMinPotentialEnergy = {ActualMinPotentialEnergy-offset}\n')

    print("\nResults for the CRO optimization\n")


    #CRO optimization algorithm
    deco = 0
    unicol = 0
    syn = 0
    intcol = 0
    tempArray = np.arange(0.25,0.4,0.01)


    for cro in range(N):
        b = np.random.rand() #random number for MoleColl
        if b > MoleColl: #80% unimolecular, #20% intermolecular
            molWeight = 1-np.exp(-(PotentialEnergy))
            molProb = molWeight/np.sum(molWeight)
            molIndex = np.arange(PopSize)
            i = np.random.choice(molIndex,p = molProb)
            mol = w[i,:] #select a random molecule
            if NumHit[i] - MinHit[i] > alfa: #decomposition trigger
                deco = deco + 1
                #--------------------------------Decomposition----------------------------------------
                #create the new molecules
                m1,m2,PotentialEnergy1,PotentialEnergy2 = operator_decomposition(mol,i)#store the new molecules and its potential energies
                #Decomposition condition
                if PotentialEnergy[i] + KineticEnergy[i] >= PotentialEnergy1 + PotentialEnergy2: #accept decomposition
                    Edec = PotentialEnergy[i] + KineticEnergy[i] - (PotentialEnergy1 + PotentialEnergy2) #cal edec
                    gamma3 = np.random.rand() #generate gamma 3
                    KineticEnergy1 = Edec * gamma3 # calculate the kinetic energy for molecule 1
                    KineticEnergy2 = Edec * (1-gamma3) #calculte the kinetic energy for molecule 2
                    MinStruct1 = m1 #store the minimum structure for molecule 1
                    MinStruct2 = m2 #store the minimum structure for molecule 2
                    MinPotentialEnergy1 = PotentialEnergy1 #store the minimum potential energy for molecule 1
                    MinPotentialEnergy2 = PotentialEnergy2 #store the minimum potential energy for molecule 2
                    #Destroy the old molecule
                    w[i,:] = m1  #index molecule 1 where the old molecule where
                    w = np.insert(w,i+1,m2,axis = 0) #index molecule 2 on the right side of molecule 1
                    #update KineticEnergy,PotentialEnergy,numHit,MinStruct,MinPotentialEnergy,MinHit for the new molecules
                    KineticEnergy[i] = KineticEnergy1
                    KineticEnergy = np.insert(KineticEnergy,i+1,KineticEnergy2)
                    PotentialEnergy[i] = PotentialEnergy1
                    PotentialEnergy = np.insert(PotentialEnergy,i+1,PotentialEnergy2)
                    #set on 0 the collisions for the new molecules
                    NumHit[i] = 0
                    NumHit = np.insert(NumHit,i+1,0)
                    #set the minimum structure and minPotentialEnergy for both new molecules
                    MinStruct[i,:] = MinStruct1
                    MinStruct = np.insert(MinStruct,i+1,MinStruct2,axis = 0)
                    MinPotentialEnergy[i] = MinPotentialEnergy1
                    MinPotentialEnergy = np.insert(MinPotentialEnergy,i+1,MinPotentialEnergy2)
                    #set on 0 the collisions min hit for each molecule
                    MinHit[i] = 0
                    MinHit = np.insert(MinHit,i+1,0)
                    #Increase the PopSize
                    PopSize = PopSize + 1

                else:
                    gamma1 = np.random.rand() #generte gamma values for grab buffer energy
                    gamma2 = np.random.rand()
                    Edec = PotentialEnergy[i] + KineticEnergy[i] + (gamma1*gamma2*buffer)-(PotentialEnergy1 - PotentialEnergy2) #calculte decomoposition energy
                    if Edec >= 0: #verify condition for decomposition
                        buffer = buffer*(1-gamma1*gamma2)
                        gamma3 = np.random.rand() #generate random gamma 3
                        #RePotentialEnergyat the process for accepting a collision
                        KineticEnergy1 = Edec * gamma3 # calculate KineticEnergy for molecule 1
                        KineticEnergy2 = Edec * (1-gamma3) #calculate KineticEnergy for molecule 2
                        MinStruct1 = m1 #store minimum structure for molecule 1
                        MinStruct2 = m2 #store minimum structure for molecule 2
                        MinPotentialEnergy1 = PotentialEnergy1 #store minimim potential energy for molecule 1
                        MinPotentialEnergy2 = PotentialEnergy2 #store minimum potential energy for molecule 2
                        #Destroy the old molecule
                        w[i,:] = m1  #index molecule 1 where the old molecule where
                        w = np.insert(w,i+1,m2,axis = 0) #index molecule 2 on the right side of molecule 1
                        #update KineticEnergy,PotentialEnergy,numHit,MinStruct,MinPotentialEnergy,MinHit for the new molecules
                        KineticEnergy[i] = KineticEnergy1
                        KineticEnergy = np.insert(KineticEnergy,i+1,KineticEnergy2)
                        PotentialEnergy[i] = PotentialEnergy1
                        PotentialEnergy = np.insert(PotentialEnergy,i+1,PotentialEnergy2)
                        #set on 0 the collisions for the new molecules
                        NumHit[i] = 0
                        NumHit = np.insert(NumHit,i+1,0)
                        #set the minimum structure and minPotentialEnergy for both new molecules
                        MinStruct[i,:] = MinStruct1
                        MinStruct = np.insert(MinStruct,i+1,MinStruct2,axis = 0)
                        MinPotentialEnergy[i] = MinPotentialEnergy1
                        MinPotentialEnergy = np.insert(MinPotentialEnergy,i+1,MinPotentialEnergy2)
                        #set on 0 the collisions min hit for each molecule
                        MinHit[i] = 0
                        MinHit = np.insert(MinHit,i+1,0)
                        #Increase the PopSize
                        PopSize = PopSize + 1
                    else:
                        #Destroy the new molecules and increase the collision for the molecule in 1
                        NumHit[i] = NumHit[i] + 1
            else:
                #--------------------------On wall ineffective collision---------------------------------------
                unicol = unicol + 1
                w_prime,PotentialEnergy_prime = operator_unicollision(mol) #apply the operator for the molecule and its energy
                NumHit[i] = NumHit[i] + 1 #increase the number of collisions in 1
                if PotentialEnergy[i] + KineticEnergy[i] >= PotentialEnergy_prime:  #verifyenergy conservation
                    a = np.random.uniform(KineticEnergyLossRate,1) #generate a random number for the kinetic energy loss
                    KineticEnergy_prime = (PotentialEnergy[i] - PotentialEnergy_prime + KineticEnergy[i])*a #calculate kinetic energy after collision
                    buffer = buffer + (PotentialEnergy[i] - PotentialEnergy_prime + KineticEnergy[i])* (1-a) #calculate the energy that goes to the buffer
                    w[i,:] = w_prime #change the old molecule for the new molecule
                    PotentialEnergy[i] = PotentialEnergy_prime #change the new potential energy
                    KineticEnergy[i] = KineticEnergy_prime #change the the kinetic energy of the molecule
                    if PotentialEnergy[i] < MinPotentialEnergy[i]: #minimize the potential energy
                        MinStruct[i:,] = w[i:,] #store the minimum structure if the potential energy
                        MinPotentialEnergy[i] = PotentialEnergy[i] #store the minimum potential energy
                        MinHit[i] = NumHit[i] #store the number of hits when a molecule reach its minimum structure

        else:
            if PopSize > 1: #its intermolecular only if the PopSize is more than one
                molWeight = 1-np.exp(-(PotentialEnergy))
                molProb = molWeight/np.sum(molWeight)
                molIndex = np.arange(PopSize)
                i = np.random.randint(PopSize)
                j = np.random.randint(PopSize)
                while i == j:
                    i = np.random.choice(molIndex,p = molProb)
                    j = np.random.choice(molIndex,p = molProb)
                m1 = w[i,:]
                m2 = w[j,:]
                if KineticEnergy[i] and KineticEnergy[j] <= beta:
                    syn = syn + 1
                    #-----------------Synthesis---------------------------------
                    m_prime,PotentialEnergy_prime = operator_synthesis(m1, m2, i, j) #Obtain the new molecule and its potential energy
                    if (PotentialEnergy[i] + PotentialEnergy[j] + KineticEnergy[i] + KineticEnergy[j])  >= PotentialEnergy_prime: #check the energy conservation
                        KineticEnergy_prime = PotentialEnergy[i] + PotentialEnergy[j] + KineticEnergy[i] + KineticEnergy[j] - PotentialEnergy_prime #calculate the kinetic energy for the new molecule
                        #accept the synthesis and replace KineticEnergy, MinStruct and MinPotentialEnergy
                        KineticEnergy[i] = KineticEnergy_prime #update the kinetic energy
                        MinStruct[i,:] = m_prime #update the minimum structure
                        MinPotentialEnergy[i] = PotentialEnergy_prime #update the minimum potential energy
                        #Destroy molecule 1 by replacing the old values with the new molecue
                        w[i,:] = m_prime #replace molecule i with the new molecule
                        PotentialEnergy[i] = PotentialEnergy_prime #replace molecule i PotentialEnergy withe the new molecule PotentialEnergy
                        NumHit[i] = 0 #set the new molecule colisiions in 0
                        MinHit[i] = 0 #set the new molecule MinHIt in 0
                        #Destroy the second molecule with python delete function
                        w = np.delete(w,j,axis = 0)
                        PotentialEnergy = np.delete(PotentialEnergy,j)
                        KineticEnergy = np.delete(KineticEnergy,j)
                        MinStruct = np.delete(MinStruct,j,axis = 0)
                        MinPotentialEnergy = np.delete(MinPotentialEnergy,j)
                        NumHit = np.delete(NumHit,j)
                        MinHit = np.delete(MinHit,j)
                        PopSize = PopSize - 1 #Decrease the population by 1

                    else:
                        #Reject the synthesis
                        NumHit[i] = NumHit[i] + 1 #update the number of collisions by 1
                        NumHit[j] = NumHit[j] + 1 #update the number of collisions by 1
                else:
                    intcol = intcol + 1
                    #----------------Intermolecular ineffective collision--------------------------
                    m1_prime,m2_prime,PotentialEnergy1_prime,PotentialEnergy2_prime = operator_collision(m1,m2) #obtain molecules 1 and 2 with their energy
                    NumHit[i] = NumHit[i] + 1  #Increase the number of collisions for molecule 1
                    NumHit[j] = NumHit[j] + 1 #Increase the number of collisions for molecule 2
                    Einter = (PotentialEnergy[i] + PotentialEnergy[j] + KineticEnergy[i] + KineticEnergy[j]) - (PotentialEnergy1_prime + PotentialEnergy2_prime) #Calculate the energy of collision
                    if Einter >= 0: #condition for accepting the intermolecular ineffective collision
                        gamma4 = np.random.rand() #generate gamma4 randomly
                        w[i,:] = m1_prime #update molecule 1
                        w[j,:] = m2_prime #update molecule 2
                        PotentialEnergy[i] = PotentialEnergy1_prime #update potential energy for molecule 1
                        PotentialEnergy[j] = PotentialEnergy2_prime #update potential energy for molecule 2
                        KineticEnergy[i] = Einter*gamma4 #update kinetic energy for molecule 1
                        KineticEnergy[j] = Einter*(1-gamma4)#update kinetic energy for molecule 2
                        #Molecule 1
                        if PotentialEnergy[i] < MinPotentialEnergy[i]:
                            MinStruct[i,:] = w[i,:] #minimize the structure
                            MinPotentialEnergy[i] = PotentialEnergy[i] #minimize the potential energy
                            MinHit[i] = NumHit[i] #register the number of collisions
                        #Molecule 2
                        if PotentialEnergy[j] < MinPotentialEnergy[j]: #Condicion de minimizacion de la energia potencial
                            MinStruct[j,:] = w[j,:] #minimize the structure
                            MinPotentialEnergy[j] = PotentialEnergy[j] #minimize the potential energy
                            MinHit[j] = NumHit[j] #register the number of collisions

        if MinPotentialEnergy.min() < ActualMinPotentialEnergy: #Check for any new minimum solution
            ActualMinPotentialEnergy = MinPotentialEnergy.min()
            MinIndex = np.where(MinPotentialEnergy == ActualMinPotentialEnergy)
            MinIndex = MinIndex[0]
            ActualMinStruct = MinStruct[MinIndex]
        print(cro,ActualMinPotentialEnergy-offset,PopSize,buffer)

        comparator = ActualMinPotentialEnergy - offset
        if round(comparator,7) == ReportedValue:
            Iterations_axis[indexAxis] = cro + 1
            break
        else:
            Iterations_axis[indexAxis] = cro + 1



    ActualMinStruct = np.resize(ActualMinStruct,[atoms,3])
    pDeco = deco*100/N
    pUnicol = unicol*100/N
    pSyn = syn*100/N
    pIntcol = intcol*100/N
    et = time.time()
    res = et - st #print the time used for optimization
    resmin = res/60 #change to minutes
    reshour = resmin/60
    #Create txt files for backups
    filestruct = open(f"lj{atoms}_xyz_{round(EaDMC,2)}.txt","w")
    filestruct.write(str(ActualMinPotentialEnergy-offset) + "\n")
    filestruct.write(str(ActualMinStruct) + "\n")
    filestruct.close()
    fileinfo = open(f"info{atoms}_{round(EaDMC,2)}.txt","w")
    fileinfo.write(f"Iterations = {N}\n")
    fileinfo.write(f"FinalPopSize = {PopSize} \n")
    fileinfo.write(f"Execution time = {reshour} hours\n")
    fileinfo.write(f"Deco = {pDeco}\n")
    fileinfo.write(f"UniCol = {pUnicol}\n")
    fileinfo.write(f"Syn = {pSyn}\n")
    fileinfo.write(f"IntCol = {pIntcol}\n")
    fileinfo.write(f"Alfa = {alfa}\n")
    fileinfo.write(f"Beta = {beta}\n")
    fileinfo.close()
    MinPE_axis[indexAxis] = ActualMinPotentialEnergy - offset


plt.figure(1)
plt.plot(EaDMC_axis,MinPE_axis,'-o')
plt.xlabel('EaDMC')
plt.ylabel('Mininimum Potential Energy')
plt.title('38 atoms LJ cluster optimization using 10000 CRO iterations')
plt.savefig('EnergyEADMC.png',dpi =300)
plt.figure(2)
plt.plot(EaDMC_axis,Iterations_axis,'-o')
plt.xlabel('EaDMC')
plt.ylabel('Iterations')
plt.title('Number of iterations that 38 atoms LJ cluster use to converge to the reported value')
plt.savefig('IterationsEADMC.png', dpi = 300)







