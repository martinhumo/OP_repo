import numpy as np
from ovito.io import import_file
from tqdm import tqdm

class Trajectory:
    def __init__(self, filename, attr, skip, fcx):
        """
        filename         : path to the trajectory file with sorted and format = id mol x y z fx fy fz vx vy vz
        attr             : particles attributes to be loaded in memory = coordinates(0), forces(1) or velocities(2)
        skip             : number of snapshots to be skipped between two configurations that are evaluated
                           (for example, if trajectory is 9000 steps long, and skip = 10, every tenth step
                           is evaluated, 900 steps in total; use skip = 1 to take every step of the MD)
        fcx              : conversion factor to have positions in [m]
         """
        pipeline = import_file(filename) #import file
        self.n_atoms = pipeline.compute().particles.count #number of atoms
        self.n_steps_total = pipeline.source.num_frames

        self.skip = skip
        self.fcx = fcx
        self.n_steps = self.n_steps_total // self.skip

        attributes = ('coordinates', 'forces', 'velocities')
        print('Are going to be loaded: particles {}'.format(attributes[attr]))
        print('Trajectory frames= ',self.n_steps_total)
        print('Frames to be loaded= ',self.n_steps)


        self.coordinates = np.empty((self.n_steps, self.n_atoms, 3))
        if attr == 0:
            self.boxsize = np.empty((self.n_steps, 3, 2))
        count = 0
        stop = self.n_steps
        for step in tqdm(range(self.n_steps)):
            frame = step * self.skip
            try:
                if attr == 0:
                    self.coordinates[step] = pipeline.compute(frame).particles.positions*fcx
                    self.boxsize[step,:,0] = pipeline.compute(frame).cell[:,3]*fcx
                    self.boxsize[step,:,1] = np.sum(pipeline.compute(frame).cell[:,:], axis=1)*fcx
                elif attr == 1:
                    self.coordinates[step] = pipeline.compute(frame).particles.forces
                elif attr == 2:
                    self.coordinates[step] = pipeline.compute(frame).particles.velocities
            except:
                if count == 0:
                    stop = step
                    print( 'file broken in step: ',step * self.skip)
                count += 1
                break
        self.coordinates =  self.coordinates[:stop,:,:]
        if attr == 0:
            self.boxsize = self.boxsize[:stop,:,:]



    def compute_orientational_order_tensor(self, nb):
        """This function take the trajectory and returns the the nematic order
        parameter and the vector director.
        Returns: NmOP [], vx[], vy[], vz[]
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize: box [m]
        *nb: atoms per molecule []

        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]"""

        print('--------------------')
        print('Computing orientational_order_tensor ')


        nm = self.n_atoms//nb #molecules number
        bondvectors = np.zeros((self.n_steps, nm*(nb-1), 3)) #director vector of bond in each step for the whole system

        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = data_box[0,1] - data_box[0,0]
            B = data_box[1,1] - data_box[1,0]
            C = data_box[2,1] - data_box[2,0]
            box = np.array([A,B,C])*0.5


            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            j=0
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.empty(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):
                    dist = (molecule[i+1]-moleculeu[i])
                    for dim in (0,1,2):
                        if dist[dim]>box[dim]:
                            dist[dim] = dist[dim] - box[dim]*2.
                            moleculeu[i+1,dim] = molecule[i+1,dim] - box[dim]*2.
                        elif dist[dim]<=(-box[dim]):
                            dist[dim] = dist[dim] + box[dim]*2.
                            moleculeu[i+1,dim] = molecule[i+1,dim] + box[dim]*2.
                        else:
                            moleculeu[i+1,dim] = molecule[i+1,dim]
                    bondvector = np.array(dist)
                    bondvector /= np.linalg.norm(bondvector)
                    bondvectors[step,j ,:] =  bondvector
                    j+=1

        Q =  np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, :, i] * bondvectors[:, :, i], axis=(0, 1)) - 1)
                else:
                    Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, :, i] * bondvectors[:, :, j], axis=(0, 1)))


        eig , eigv = np.linalg.eig(Q)
        idx = eig.argsort()[::-1] #highest to lowest
        eig = eig[idx]
        eigv = eigv[:,idx]
        return eig[0], eigv[:,0].T

    def compute_orientational_order_tensor_evolution(self, nb):
        """This function take the trajectory and returns the the nematic order
        parameter and the vector director for each frame.
        Returns: step[:n_steps,], [:n_steps,], NmOP [:n_steps,], vx[:n_steps,], vy[:n_steps,], vz[:n_steps,]
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize: box [m]
        *nb: atoms per molecule []

        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]"""

        print('--------------------')
        print('Computing orientational_order_tensor ')


        nm = self.n_atoms//nb #molecules number
        bondvectors = np.zeros(( nm*(nb-1), 3)) #director vector of bond in each step for the whole system
        S2_evol = np.zeros((self.n_steps, 5))

        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = data_box[0,1] - data_box[0,0]
            B = data_box[1,1] - data_box[1,0]
            C = data_box[2,1] - data_box[2,0]
            box = np.array([A,B,C])*0.5


            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            j=0
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.empty(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):
                    dist = (molecule[i+1]-moleculeu[i])
                    for dim in (0,1,2):
                        if dist[dim]>box[dim]:
                            dist[dim] = dist[dim] - box[dim]*2.
                            moleculeu[i+1,dim] = molecule[i+1,dim] - box[dim]*2.
                        elif dist[dim]<=(-box[dim]):
                            dist[dim] = dist[dim] + box[dim]*2.
                            moleculeu[i+1,dim] = molecule[i+1,dim] + box[dim]*2.
                        else:
                            moleculeu[i+1,dim] = molecule[i+1,dim]
                    bondvector = np.array(dist)
                    bondvector /= np.linalg.norm(bondvector)
                    bondvectors[j ,:] =  bondvector
                    j+=1

            Q =  np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    if i == j:
                        Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, i] * bondvectors[:, i], axis=0) - 1)
                    else:
                        Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, i] * bondvectors[:, j], axis=0))


            eig , eigv = np.linalg.eig(Q)
            idx = eig.argsort()[::-1] #highest to lowest
            eig = eig[idx]
            eigv = eigv[:,idx]
            frame = step * self.skip
            S2_evol[step,:] = np.concatenate(([frame, eig[0]],  eigv[:,0]), axis=None)

        return S2_evol
    
    def compute_orientational_order_tensor_evolution_tiberio(self, nb):
        """This function take the trajectory and returns the the nematic order
        parameter and the vector director for each frame.
        Returns: step[:n_steps,], [:n_steps,], NmOP [:n_steps,], vx[:n_steps,], vy[:n_steps,], vz[:n_steps,]
        Needs:
        *self.coordinates: atoms positions [a.u.]
        *self.boxsize:
        *nb: atoms per molecule []

        References:
        (1) An atomistic simulation for 4-cyano-4′-pentylbiphenyl and its homologue with a reoptimized force field. [2011] https://doi.org/10.1021/jp111408n
        
        """

        print('--------------------')
        print('Computing orientational_order_tensor ')


        nm = self.n_atoms//nb #molecules number
        bondvectors = np.zeros(( nm, 3)) #director vector of bond in each step for the whole system
        S2_evol = np.zeros((self.n_steps, 5))

        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = data_box[0,1] - data_box[0,0]
            B = data_box[1,1] - data_box[1,0]
            C = data_box[2,1] - data_box[2,0]
            box = np.array([A,B,C])*0.5


            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.empty(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):
                    dist = (molecule[i+1]-moleculeu[i])
                    for xyz in (0,1,2):
                        if dist[xyz]>box[xyz]:
                            dist[xyz] = dist[xyz] - box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                        elif dist[xyz]<=(-box[xyz]):
                            dist[xyz] = dist[xyz] + box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
                direct = moleculeu[0,:] - moleculeu[13,:]    #direction between N and C12 #Fig 2 Tiberio 2009
                bondvector = np.array(direct)
                bondvector /= np.linalg.norm(bondvector)
                bondvectors[mol-1 ,:] =  bondvector


            Q =  np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    if i == j:
                        Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, i] * bondvectors[:, i], axis=0) - 1)
                    else:
                        Q[i, j] = 0.5 * (np.mean(3 * bondvectors[:, i] * bondvectors[:, j], axis=0))


            eig , eigv = np.linalg.eig(Q)
            idx = eig.argsort()[::-1] #highest to lowest
            eig = eig[idx]
            eigv = eigv[:,idx]
            frame = step * self.skip
            S2_evol[step,:] = np.concatenate(([frame, eig[0]],  eigv[:,0]), axis=None)

        return S2_evol


    def compute_structure_factor(self, nb, m):
        """This function take the trajectory and returns the structure factor
         SmAOP (SmA Order Parameter) as the value of the highest peak .
        Returns:  SmAOP [Beads diameter]
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize:
        *nb: atoms per molecule []
        *m: mass of the atoms [kg]


        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]
        (2) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]
        (3) Numerical study of the phase behavior of rod-like colloidal particles with attractive tips [2021]
        """

        print('--------------------')
        print('Computing Structure Factor ')


        nm = self.n_atoms//nb #molecules number
        Qz =  np.linspace( 0.25, 3.0, 10000)*(1. / self.fcx) #Qz = 2pi/delta , delta = layer thickness
        Ss = np.zeros((self.n_steps, len(Qz))) #Structure factor for step

        Zp = np.zeros(nm) #Atoms Z component for step

        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = data_box[0,1] - data_box[0,0]
            B = data_box[1,1] - data_box[1,0]
            C = data_box[2,1] - data_box[2,0]
            box = np.array([A,B,C])*0.5

            data_beads = np.array(self.coordinates[step])#step atoms cordinates

            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):                #p.b.c. in z direction
                    dist_z = (molecule[i+1,2]-moleculeu[i,2])
                    if dist_z > box[2]:
                        moleculeu[i+1,2] = molecule[i+1,2] - box[2]*2.
                    elif dist_z <= (-box[2]):
                        moleculeu[i+1,2] = molecule[i+1,2] + box[2]*2.
                    else:
                        moleculeu[i+1,2] = molecule[i+1,2]

                Zp[(mol-1)] =np.average(moleculeu, axis=0, weights=m)[2]  #z component of each particle(molecule mass center)

            Ss[step, :] = np.abs(np.sum(np.exp( np.outer(Zp,Qz*1j)),axis=0)) #VERRR PAPER : Naderi, S., & van der Schoot, P. (2014). Effect of bending flexibility on the phase behavior and dynamics of rods. The Journal of Chemical Physics, 141(12), 124901. doi:10.1063/1.4895730
                                                                          #Ver Paper: maMilchev, A., Nikoubashman, A., & Binder, K. (2019). The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study. Computational Materials Science, 166(May), 230–239. https://doi.org/10.1016/j.commatsci.2019.04.017
        S = (1/nm)*np.mean(Ss,axis=0) #Structure factor in function of Qz
        return np.max(S)
    
    def compute_structure_factor_evolution(self, nb, m):
        """This function take the trajectory and returns the structure factor
         SmAOP (SmA Order Parameter) as the value of the highest peak .
        Returns:  SmAOP (SmA temporal evolution ) [Beads diameter]
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize:
        *nb: atoms per molecule []
        *m: mass of the atoms [kg]


        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]
        (2) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]
        (3) Numerical study of the phase behavior of rod-like colloidal particles with attractive tips [2021]
        """

        print('--------------------')
        print('Computing Structure Factor ')


        nm = self.n_atoms//nb #molecules number
        Qz =  np.linspace( 0.25, 3.0, 10000)*(1. / self.fcx) #Qz = 2pi/delta , delta = layer thickness
        SmAOP_evol = np.zeros((self.n_steps, 2))

        Zp = np.zeros(nm) #Atoms Z component for step

        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = data_box[0,1] - data_box[0,0]
            B = data_box[1,1] - data_box[1,0]
            C = data_box[2,1] - data_box[2,0]
            box = np.array([A,B,C])*0.5

            data_beads = np.array(self.coordinates[step])#step atoms cordinates

            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]
                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):                #p.b.c. in z direction
                    dist_z = (molecule[i+1,2]-moleculeu[i,2])
                    if dist_z > box[2]:
                        moleculeu[i+1,2] = molecule[i+1,2] - box[2]*2.
                    elif dist_z <= (-box[2]):
                        moleculeu[i+1,2] = molecule[i+1,2] + box[2]*2.
                    else:
                        moleculeu[i+1,2] = molecule[i+1,2]

                Zp[(mol-1)] =np.average(moleculeu, axis=0, weights=m)[2]  #z component of each particle(molecule mass center)

            Ss= np.abs(np.sum(np.exp( np.outer(Zp,Qz*1j)),axis=0)) #VERRR PAPER : Naderi, S., & van der Schoot, P. (2014). Effect of bending flexibility on the phase behavior and dynamics of rods. The Journal of Chemical Physics, 141(12), 124901. doi:10.1063/1.4895730
            frame = step * self.skip
            SmAOP = np.max((1 / nm) * Ss)
            SmAOP_evol[step, :] = np.array([frame, SmAOP])
                                                                          #Ver Paper: maMilchev, A., Nikoubashman, A., & Binder, K. (2019). The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study. Computational Materials Science, 166(May), 230–239. https://doi.org/10.1016/j.commatsci.2019.04.017
        return SmAOP_evol
    
        
    def compute_hexatic_bond_order(self, nb, m, Sigma):
        """This function take the trajectory and returns the hexatic bond order parameter
        (H6OP)  .
        Returns:  H6OP [] mean over each step
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize: box size [m]
        *nb: atoms per molecule []
        *m: mass of the atoms [kg]
        *Sigma: particle diameter [m].


        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study. [2019]
        (2) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]
        """

        print('--------------------')
        print('Computing Hexatic Bond Order ')

        nm = self.n_atoms//nb #molecules number
        L = (nb-1)*Sigma/2 + Sigma

        hexatic_bond_order = np.zeros((self.n_steps, nm), dtype=complex)
        for step in range(self.n_steps):
            data_box = np.array(self.boxsize[step])  # Step box size
            box = (data_box[:, 1] - data_box[:, 0]) * 0.5

            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            mass_centers_unwrapped = [] #frame molecule mass centers
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                flag = False
                for i in range(nb-1):                #p.b.c. in x y z direction
                    for xyz in range(3):
                        dist_z = (molecule[i+1,xyz]-moleculeu[i,xyz])
                        if dist_z > box[xyz]:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                            flag = True
                        elif dist_z <= (-box[xyz]):
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                            flag = True
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
                mol_mass_center = np.average(moleculeu, axis=0, weights=m)
                mass_centers_unwrapped.append(mol_mass_center)

            #loop over each mass center
            for i in range(len(mass_centers_unwrapped)):
                mol_i = mass_centers_unwrapped[i].copy()
                # loop over j != i
                exp_i = 0
                nearest_neighbors = 0
                for j, mol_j in enumerate(mass_centers_unwrapped):
                    if i != j:
                        # check distance in x y z direction for periodic boundary conditions, if true move the mass center
                        dist = mol_j - mol_i
                        for dim in range(3):
                            if dist[dim] > box[dim]:
                                mol_j[dim] -= box[dim] * 2.
                            elif dist[dim] <= -box[dim]:
                                mol_j[dim] += box[dim] * 2.

                        #calculate xy vector
                        vector_xy = mol_j[:2] - mol_i[:2]

                        #calculate distance in z and xy
                        dist_xy = np.linalg.norm(vector_xy)
                        dist_z = abs(mol_j[2] - mol_i[2])

                        #calculate hexatic bond order
                        if dist_xy <= 1.7*Sigma and dist_z <= L/2:
                            #calculate angle between vector_xy and x axis (1,0)
                            cos_phi_i_j = np.dot(vector_xy / dist_xy , np.array([1, 0])) #/ (1.0 * 1.0)  #Dot product(v1, v2) / |v1| |v2|
                            phi_i_j = np.arccos(cos_phi_i_j)

                            exp_i_j = np.exp(6j*phi_i_j)
                            exp_i += exp_i_j
                            nearest_neighbors += 1

                    else:
                        continue
                #Sum the contribution of i to hexatic bond order (revisar porque para cada molecula no se suma)
                if nearest_neighbors != 0:
                    hexatic_bond_order[step, i] += exp_i / nearest_neighbors

        #Average real part and imaginary part in each step
        hexatic_bond_order_real = np.mean(np.real(hexatic_bond_order), axis=1)
        hexatic_bond_order_imag = np.mean(np.imag(hexatic_bond_order), axis=1)

        #Absolute value of hexatic bond order
        hexatic_bond_order = np.sqrt(hexatic_bond_order_real**2 + hexatic_bond_order_imag**2) #See paper (2)

        return  np.mean(hexatic_bond_order)
    
    def compute_hexatic_bond_order_evolution(self, nb, m, Sigma):
        """This function take the trajectory and returns the hexatic bond order parameter
        (H6OP)  temporal evolution.
        Returns:  H6OP [] and frames.
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize: box size [m]
        *nb: atoms per molecule []
        *m: mass of the atoms [kg]
        *Sigma: particle diameter [m].


        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study. [2019]
        (2) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]
        """

        print('--------------------')
        print('Computing Hexatic Bond Order ')

        nm = self.n_atoms//nb #molecules number
        L = (nb-1)*Sigma/2 + Sigma

        hexatic_bond_order = np.zeros((self.n_steps, nm), dtype=complex)
        frames = np.zeros(self.n_steps)
        for step in range(self.n_steps):
            data_box = np.array(self.boxsize[step])  # Step box size
            box = (data_box[:, 1] - data_box[:, 0]) * 0.5

            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            mass_centers_unwrapped = [] #frame molecule mass centers
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                flag = False
                for i in range(nb-1):                #p.b.c. in x y z direction
                    for xyz in range(3):
                        dist_z = (molecule[i+1,xyz]-moleculeu[i,xyz])
                        if dist_z > box[xyz]:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                            flag = True
                        elif dist_z <= (-box[xyz]):
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                            flag = True
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
                mol_mass_center = np.average(moleculeu, axis=0, weights=m)
                mass_centers_unwrapped.append(mol_mass_center)

            #loop over each mass center
            for i in range(len(mass_centers_unwrapped)):
                mol_i = mass_centers_unwrapped[i].copy()
                # loop over j != i
                exp_i = 0
                nearest_neighbors = 0
                for j, mol_j in enumerate(mass_centers_unwrapped):
                    if i != j:
                        # check distance in x y z direction for periodic boundary conditions, if true move the mass center
                        dist = mol_j - mol_i
                        for dim in range(3):
                            if dist[dim] > box[dim]:
                                mol_j[dim] -= box[dim] * 2.
                            elif dist[dim] <= -box[dim]:
                                mol_j[dim] += box[dim] * 2.

                        #calculate xy vector
                        vector_xy = mol_j[:2] - mol_i[:2]

                        #calculate distance in z and xy
                        dist_xy = np.linalg.norm(vector_xy)
                        dist_z = abs(mol_j[2] - mol_i[2])

                        #calculate hexatic bond order
                        if dist_xy <= 1.7*Sigma and dist_z <= L/2:
                            #calculate angle between vector_xy and x axis (1,0)
                            cos_phi_i_j = np.dot(vector_xy / dist_xy , np.array([1, 0])) #/ (1.0 * 1.0)  #Dot product(v1, v2) / |v1| |v2|
                            phi_i_j = np.arccos(cos_phi_i_j)

                            exp_i_j = np.exp(6j*phi_i_j)
                            exp_i += exp_i_j
                            nearest_neighbors += 1

                    else:
                        continue
                #Sum the contribution of i to hexatic bond order 
                if nearest_neighbors != 0:
                    hexatic_bond_order[step, i] += exp_i / nearest_neighbors
            frame = step * self.skip
            frames[step] = frame

        #Average real part and imaginary part in each step
        hexatic_bond_order_real = np.mean(np.real(hexatic_bond_order), axis=1)
        hexatic_bond_order_imag = np.mean(np.imag(hexatic_bond_order), axis=1)

        #Absolute value of hexatic bond order
        Phi6 = np.sqrt(hexatic_bond_order_real**2 + hexatic_bond_order_imag**2) #See paper (2)

        #concatenate Phi6 and frames
        Phi6_evol = np.column_stack((frames, Phi6))
         
        return  Phi6_evol
    

    def mc_pair_correlation_function(self, nb, m, Sigma, resolution=200):
        """This function take the trajectory and returns the mean mass center pair correlation function (MPCF)  .
        Returns: bins_distances [] array, MPCF [] array mean over all steps, StdMPCF [] array 
        Needs:
        *self.coordinates: atoms positions [m]
        *self.boxsize: box size [m]
        *nb: atoms per molecule []
        *m: mass of the atoms [kg]
        *Sigma: particle diameter [m]
        *resolution: number of bins for the correlation function.

        References:
        (1) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]
        (2) https://github.com/aberut/paircorrelation2d/blob/master/paircorrelation2d.py
        (3) https://github.com/ggosti/PairCorrelationFunction/blob/main/pairCorrelationFunction.py
        """

        print('--------------------')
        print('Computing MPCF...')

        nm = self.n_atoms//nb #molecules number
        L = (nb-1)*Sigma/2 + Sigma

        mc_pair_correlations = np.zeros((self.n_steps, resolution-1))
        bins_distances = np.linspace(0.0, L, resolution) #array of bins for the correlation function

        for step in range(self.n_steps):
            data_box = np.array(self.boxsize[step])  # Step box size
            box = (data_box[:, 1] - data_box[:, 0]) * 0.5
            rho = nm / ((box[0] * box[1] * box[2]) * 2.)

            data_beads = np.array(self.coordinates[step])#step atoms cordinates
            mass_centers_unwrapped = [] #frame molecule mass centers
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
  
                for i in range(nb-1):                #p.b.c. in x y z direction
                    for xyz in range(3):
                        dist_z = (molecule[i+1,xyz]-moleculeu[i,xyz])
                        if dist_z > box[xyz]:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                        elif dist_z <= (-box[xyz]):
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
                mol_mass_center = np.average(moleculeu, axis=0, weights=m)
                mass_centers_unwrapped.append(mol_mass_center)
            mass_centers_unwrapped = np.array(mass_centers_unwrapped)

            #Correlation function    
            g_of_r=np.zeros(len(bins_distances)-1) #array of zeros for the correlation function
            for ii in range(nm):
                #coordinates of the current point
                x_loc=mass_centers_unwrapped[ii,0]
                y_loc=mass_centers_unwrapped[ii,1]
                z_loc=mass_centers_unwrapped[ii,2]
                
                dist_to_loc_xy=np.sqrt((mass_centers_unwrapped[:,0]-x_loc)**2+
                                    (mass_centers_unwrapped[:,1]-y_loc)**2) #distance of the current point to each other point
                dist_to_loc_xy[ii]=np.inf #the distance from a point to itself is always zero (it is therefore excluded from the computation)
                dist_to_loc_z = np.sqrt((mass_centers_unwrapped[:,2]-z_loc)**2) #distance of the current point to each other point
                dist_to_loc_z[ii] = np.inf

                dist_to_loc_layer = dist_to_loc_xy[dist_to_loc_z <= L/2] #distance of the current point to each other point in the same layer
                
                hist, bin_edges = np.histogram(dist_to_loc_layer,bins=bins_distances) 
                deltaV =   np.pi* L * (bin_edges[1:]**2 - bin_edges[:-1]**2)
                
                g_of_r += hist / deltaV #computes the histogram of distances to the current point

            mc_pair_correlations[step] = g_of_r / rho
        return  bins_distances[1:] ,np.mean(mc_pair_correlations , axis=0), np.std(mc_pair_correlations, axis=0)
    

    import numpy as np


    def calculate_contour_length(self, nb):
        """
        Calcula la longitud de contorno promedio de moléculas lineales en la trayectoria.
        
        Needs:
            *self.coordinates: atoms positions [m]
            *self.boxsize: box [m]

        Parameters:
            nb (int): Número de átomos por molécula.

        Returns:
            float: Longitud de contorno promedio de las moléculas. [m]
            float: Desviación estándar de la longitud de contorno promedio de las moléculas. [m]
        """
        n_steps, n_atoms, _ = self.coordinates.shape
        nm = n_atoms // nb  # Número de moléculas

        contour_lengths = []
        for step in range(n_steps):
            data_box = np.array(self.boxsize[step])  # Tamaño de la caja en este paso
            box = (data_box[:, 1] - data_box[:, 0]) * 0.5
            data_beads = np.array(self.coordinates[step])  # Coordenadas de los átomos en este paso
            
            for mol in range(nm):
                molecule = data_beads[mol * nb:(mol + 1) * nb, :]  # Átomos de la molécula
                molecule_unwrapped = np.zeros_like(molecule)
                molecule_unwrapped[0, :] = molecule[0, :]
                
                # Desempaquetado para condiciones periódicas
                for i in range(1, nb):
                    for xyz in range(3):
                        dist_z = molecule[i, xyz] - molecule_unwrapped[i - 1, xyz]
                        if dist_z > box[xyz]:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz] - 2 * box[xyz]
                        elif dist_z <= -box[xyz]:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz] + 2 * box[xyz]
                        else:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz]
                
                # Calcular longitud de contorno (suma de distancias entre átomos consecutivos)
                distances = np.linalg.norm(np.diff(molecule_unwrapped, axis=0), axis=1)
                contour_lengths.append(np.sum(distances))
        
        average_contour_length = np.mean(contour_lengths)
        std_contour_length = np.std(contour_lengths)
        return average_contour_length, std_contour_length
    

    def calculate_end_to_end_length(self, nb):
        """
        Calcula la longitud end-to-end promedio de moléculas lineales en la trayectoria.
        
        Needs:
            *self.coordinates: atoms positions [m]
            *self.boxsize: box [m]

        Parameters:
            nb (int): Número de átomos por molécula.

        Returns:
            float: Longitud end-to-end promedio de las moléculas. [m]
            float: Desviación estandard de la longitud end-to-end de las moléculas. [m]
        """
        n_steps, n_atoms, _ = self.coordinates.shape
        nm = n_atoms // nb  # Número de moléculas

        end_to_end_lengths = []
        for step in range(n_steps):
            data_box = np.array(self.boxsize[step])  # Tamaño de la caja en este paso
            box = (data_box[:, 1] - data_box[:, 0]) * 0.5
            data_beads = np.array(self.coordinates[step])  # Coordenadas de los átomos en este paso
            
            for mol in range(nm):
                molecule = data_beads[mol * nb:(mol + 1) * nb, :]  # Átomos de la molécula
                molecule_unwrapped = np.zeros_like(molecule)
                molecule_unwrapped[0, :] = molecule[0, :]
                
                # Desempaquetado para condiciones periódicas
                for i in range(1, nb):
                    for xyz in range(3):
                        dist_z = molecule[i, xyz] - molecule_unwrapped[i - 1, xyz]
                        if dist_z > box[xyz]:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz] - 2 * box[xyz]
                        elif dist_z <= -box[xyz]:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz] + 2 * box[xyz]
                        else:
                            molecule_unwrapped[i, xyz] = molecule[i, xyz]
                
                # Calcular la distancia end-to-end
                end_to_end_distance = np.linalg.norm(molecule_unwrapped[-1, :] - molecule_unwrapped[0, :])
                end_to_end_lengths.append(end_to_end_distance)
        
        average_end_to_end_length = np.mean(end_to_end_lengths)
        std_end_to_end_length = np.std(end_to_end_lengths)
        return average_end_to_end_length, std_end_to_end_length


