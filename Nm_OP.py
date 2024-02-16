import numpy as np
from ovito.io import import_file


class Trajectory:
    def __init__(self, filename, attr, skip):
        """
        filename         : path to the trajectory file with sorted and format = id mol x y z fx fy fz vx vy vz
        attr             : particles attributes to be loaded in memory = coordinates(0), forces(1) or velocities(2)
        skip             : number of snapshots to be skipped between two configurations that are evaluated
                           (for example, if trajectory is 9000 steps long, and skip = 10, every tenth step
                           is evaluated, 900 steps in total; use skip = 1 to take every step of the MD)
         """
        pipeline = import_file(filename) #import file
        self.n_atoms = pipeline.compute().particles.count #number of atoms
        self.n_steps_total = pipeline.source.num_frames

        self.skip = skip
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
        for step in range(self.n_steps):
            print('Loading Frame:', step*skip,' ',end='\x1b[1K\r')
            frame = step * self.skip
            try:

                if attr == 0:
                    self.coordinates[step] = pipeline.compute(frame).particles.positions
                    self.boxsize[step,:,0] = pipeline.compute(frame).cell[:,3]
                    self.boxsize[step,:,1] = np.sum(pipeline.compute(frame).cell[:,:], axis=1)
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



    def compute_orientational_order_tensor(self, nb, fc):
        """This function take the trajectory and returns the the nematic order
        parameter and the vector director.
        Returns: NmOP [], vx[], vy[], vz[]
        Needs:
        *self.coordinates: atoms positions [a.u.]
        *self.boxsize:
        *nb: atoms per molecule []
        *fc: conversion factor to have positions in [m].

        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]"""

        print('--------------------')
        print('Computing orientational_order_tensor ')


        nm = self.n_atoms//nb #molecules number
        self.coordinates *= fc
        self.boxsize *= fc
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
                    for xyz in (0,1,2):
                        if dist[xyz]>box[xyz]:
                            dist[xyz] = dist[xyz] - box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                        elif dist[xyz]<=(-box[xyz]):
                            dist[xyz] = dist[xyz] + box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
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

    def compute_orientational_order_tensor_evolution(self, nb, fc):
        """This function take the trajectory and returns the the nematic order
        parameter and the vector director for each frame.
        Returns: step[:n_steps,], [:n_steps,], NmOP [:n_steps,], vx[:n_steps,], vy[:n_steps,], vz[:n_steps,]
        Needs:
        *self.coordinates: atoms positions [a.u.]
        *self.boxsize:
        *nb: atoms per molecule []
        *fc: conversion factor to have positions in [m].

        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]"""

        print('--------------------')
        print('Computing orientational_order_tensor ')


        nm = self.n_atoms//nb #molecules number
        self.coordinates *= fc
        self.boxsize *= fc
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
                    for xyz in (0,1,2):
                        if dist[xyz]>box[xyz]:
                            dist[xyz] = dist[xyz] - box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                        elif dist[xyz]<=(-box[xyz]):
                            dist[xyz] = dist[xyz] + box[xyz]*2.
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
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


    def compute_structure_factor(self, nb, Sigma,fc):
        """This function take the trajectory and returns the structure factor
         S(Lambda) and SmAOP (SmA Order Parameter) as the value of the highest peak .
        Returns: S(Lambda) [], SmAOP []
        Needs:
        *self.coordinates: atoms positions [a.u.]
        *self.boxsize:
        *nb: atoms per molecule []
        *Sigma: particle diameter [m].
        *fc: conversion factor to have positions in [m].

        References:
        (1) The smectic phase in semiflexible polymer materials: A large scale molecular dynamics study [2019]
        (2) On the Phase Behaviour of Semi-Flexible Rod-Like Particles [2015]"""

        print('--------------------')
        print('Computing Structure Factor ')


        nm = self.n_atoms//nb #molecules number
        L = (nb-1)*(Sigma/2) + Sigma #molecule lenght
        Qz =  np.linspace( 0.5, nb, 10*100)
        Ss = np.zeros((self.n_steps, len(Qz))) #Structure factor for step

        Zp = np.zeros(self.n_atoms) #Atoms Z component for step

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

                Zp[(mol-1)*nb:(mol*nb)] = moleculeu[:,2]

            Ss[step] = np.abs(np.sum(np.exp( np.outer(Zp,Qz*1j)),axis=0)) #VERRR PAPER : Naderi, S., & van der Schoot, P. (2014). Effect of bending flexibility on the phase behavior and dynamics of rods. The Journal of Chemical Physics, 141(12), 124901. doi:10.1063/1.4895730
        S = (1/self.n_atoms)*np.mean(Ss,axis=0)
        print('SmAOP:',np.max(S[1:]))
        from matplotlib import pyplot as plt
        plt.rc('font', family='serif')
        fig, axs = plt.subplots(1,figsize=(4, 4), dpi=240)
        axs.plot( Qz , S,'g-',marker ='1')
