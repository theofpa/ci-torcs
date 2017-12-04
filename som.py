import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

class SelfOrganizingMap:
    def __init__(self, h, w, m):
        """ A self organizing map with dimensions h by w
        and input dimensionality n. Note that a map of
        this dimensions creates h * w neurons, each with
        m weights.
        """
        self.h = h
        self.w = w
        self.m = m
        self.n_units = h * w
        self.map_radius = max(h-1, w-1)/2
        self.neurons = np.empty((m, h * w))
        self.map_pos = np.array([(i,j) for i in range(h) for j in range(w)])
        self.energy = []
        self.energy_times = []

    def self_organize(self, data, iters=1000):
        """ Carries out self-organization in the SOM using data.
        Args:
            - data: An m by N matrix with data points in the columns.
        """        
        if iters <= 0:
            raise ValueError("iters must be positive")
        if data.shape[0] != self.m:
            raise ValueError("Incorrect data dimensions")

        # Initialize neuron weights from unique samples of the data
        indices = np.random.choice([i for i in range(data.shape[1])], self.n_units, replace=False)
        #self.neurons[...] = data[:, indices] # Initialize with samples
        #self.neurons = np.random.rand(self.m, self.h*self.w) # Randomly initialize all weights between 0 and 1
        min_data = np.min(data, axis=1)[:, np.newaxis]
        max_data = np.max(data, axis=1)[:, np.newaxis]
        self.neurons[...] = np.random.random((self.m, self.n_units))*(max_data - min_data) + min_data

        self.energy = []
        
        # Algorithm constants
        r0 = self.map_radius 
        t1 = iters/np.log(self.map_radius)
        t2 = iters
        e0 = 0.5
        # Run!
        print("Self-organizing...")
        self._adapt(data, iters, r0, t1, t2, e0)       
        print("Done. Energy attained: {:.3f}".format(self.energy[-1]))

    def _adapt(self, data, n_iters, r0, t1, t2, e0):
        N = data.shape[1]
        prev_n = 0
        percents = [25, 50, 75, 90]
        for n in range(n_iters):
            done = int(100 * n/n_iters)
            
            if done in percents:
                print(str(done) + "% completed.")
                percents = percents[1:]
            # Sample data point
            i = np.random.randint(0, N)
            x = data[:,i:i+1]

            # 1) Competitive process: find coordinate of the winning neuron
            winner_i = np.unravel_index(np.argmin(np.linalg.norm(self.neurons - x, axis=0)), (self.h, self.w))

            # 2) Cooperative process: calculate neighborhood radius
            # and select neurons within it                
            radius = r0 * np.exp(-n/t1)
            distances = np.linalg.norm(self.map_pos - winner_i, axis=1)
            selected = distances <= radius
            neighborhood = np.exp(-distances[selected]**2 / (2*radius**2))
            
            # 3) Adaptive process: update weights of selected neurons
            e = e0 * np.exp(-n/t2)
            self.neurons[:, selected] -= e * neighborhood * (self.neurons[:, selected] - x)

            # The energy is sampled 10 times
            if (n - prev_n) >= n_iters/10:
                prev_n = n
                self.energy.append(self._calculate_energy(data))
        
        # Append last attained energy and calculate an epochs vector for it
        self.energy.append(self._calculate_energy(data))
        self.energy_times = np.linspace(0, 1, len(self.energy), endpoint=True) * n_iters
        
    def _calculate_energy(self, data):
        """ Calculates the energy from the data
        as the average minimum distance to the closest
        prototype.
        Args:
            - data: array, m by N
        Returns:
            - float, the average energy
        """
        average_energy = 0
        for i in range(data.shape[1]):
            x = data[:,i:i+1]
            distances = np.linalg.norm(self.neurons - x, axis=0)
            closest_i = np.argmin(distances)
            average_energy += distances[closest_i]
        return average_energy/data.shape[1]

    def plot_energy(self):
        """ Plots the history of the SOM energy during
        the self-organization process, if existent.                
        """
        if len(self.energy) > 0:
            plt.figure()
            plt.plot(self.energy_times, self.energy, "--bo")
            plt.title("Training Energy")
            plt.xlabel("Epochs")
            plt.grid()
            plt.show(block=False)
        else:
            print("No energy stored")

    def plot_hitmap(self, data):
        """ Plots the hit map of the SOM given the data.
        The area of the filled squares corresponds to the
        relative count of hits.
        Args:
            - data: array, m by N.
        """
        plt.figure()
        axis = plt.gca()

        # For each data points count the hits for the BMU
        rel_hits = np.zeros(self.n_units)
        for i in range(data.shape[1]):
            x = data[:,i:i+1]
            distances = np.linalg.norm(self.neurons - x, axis=0)
            closest_i = np.argmin(distances)
            rel_hits[closest_i] += 1
        rel_hits /= np.max(rel_hits)
        
        # Add grid and square for each neuron
        for i in range(self.h - 1, -1, -1):
            for j in range(self.w):
                axis.add_patch(Rectangle((j, i), 1, 1, fill=False))
                side = rel_hits[np.ravel_multi_index((i, j), (self.h, self.w))]
                d = (1 - side)/2
                axis.add_patch(Rectangle((j+d, i+d), side, side, alpha=0.4))
        
        # Adjust visuals and plot
        plt.title("Hit Map")
        axis.axis("off")
        plt.xlim([-0.1,self.w+0.1])
        plt.ylim([-0.1,self.h+0.1])
        axis.set_aspect('equal', 'datalim')
        plt.show(block=False)

    def plot_2dmap(self, data=None):
        """ Plots the distribution of the map in two dimensions,
        including the connections for neighbouring neurons.
        Only available if the dimensionality of the map is 2.
        Args:
            - data: (optional) an m by N array. If included, the
                    map is plotted together with the provided data.
        """
        if self.m != 2:
            raise ValueError("SOM dimensionality is not two")            

        # Plot points first
        plt.figure()
        if data is not None:
            plt.scatter(data[0,:], data[1,:], alpha=0.5, edgecolors="none")
        plt.scatter(self.neurons[0,:], self.neurons[1,:], s=10, c="k")

        # Add the lines
        axis = plt.gca()
        i = 0
        j = 0
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                # Connect to neurons in front of and below the neuron
                neuron_idx = self.neurons[:,np.ravel_multi_index((i, j), (self.h, self.w))]
                neuron_front = self.neurons[:,np.ravel_multi_index((i, j+1), (self.h, self.w))]
                neuron_below = self.neurons[:,np.ravel_multi_index((i+1, j), (self.h, self.w))]
                
                axis.add_patch(ConnectionPatch(neuron_idx, neuron_front, "data"))
                axis.add_patch(ConnectionPatch(neuron_idx, neuron_below, "data"))
            
            # For the last column, connect to neuron below
            neuron_idx = self.neurons[:,np.ravel_multi_index((i, j+1), (self.h, self.w))]
            neuron_below = self.neurons[:,np.ravel_multi_index((i+1, j+1), (self.h, self.w))]
            axis.add_patch(ConnectionPatch(neuron_idx, neuron_below, "data"))
        for j in range(self.w - 1):
            # For the last row, connect to neurons in front of the neuron
            neuron_idx = self.neurons[:,np.ravel_multi_index((self.h-1, j), (self.h, self.w))]
            neuron_front = self.neurons[:,np.ravel_multi_index((self.h-1, j+1), (self.h, self.w))]
            axis.add_patch(ConnectionPatch(neuron_idx, neuron_front, "data"))

        plt.show(block=False)


    def plot_umatrix(self):
        """ Plots a matrix with the differences among neighboring neurons.
        Based on A. Ultsch, "Self-Organizing Neural Networks for Visualisation and Classification"
        """
        umatrix = np.zeros((self.h*2 - 1, self.w*2 - 1))
        for i in range(umatrix.shape[0]):
            # For even rows, find distance between left and right neurons
            if i % 2 == 0:
                for j in range(1, umatrix.shape[1], 2):
                    back_i = np.ravel_multi_index((i//2, j//2), (self.h, self.w))
                    front_i = np.ravel_multi_index((i//2, (j+1)//2), (self.h, self.w))
                    umatrix[i, j] = np.linalg.norm(self.neurons[:, back_i] - self.neurons[:, front_i])
            # For odd rows...
            else:
                for j in range(umatrix.shape[1]):
                    # For even columns, find distance between top and bottom neurons
                    if j % 2 == 0:
                        above_i = np.ravel_multi_index(((i-1)//2, j//2), (self.h, self.w))
                        below_i = np.ravel_multi_index(((i+1)//2, j//2), (self.h, self.w))
                        umatrix[i, j] = np.linalg.norm(self.neurons[:, above_i] - self.neurons[:, below_i])
                    # For odd columns, find the midpoint between the distances of neurons in the diagonal
                    else:
                        ne_i = np.ravel_multi_index(((i-1)//2, (j+1)//2), (self.h, self.w))
                        nw_i = np.ravel_multi_index(((i-1)//2, (j-1)//2), (self.h, self.w))
                        se_i = np.ravel_multi_index(((i+1)//2, (j+1)//2), (self.h, self.w))
                        sw_i = np.ravel_multi_index(((i+1)//2, (j-1)//2), (self.h, self.w))

                        dist1 = np.linalg.norm(self.neurons[:, sw_i] - self.neurons[:, ne_i])
                        dist2 = np.linalg.norm(self.neurons[:, se_i] - self.neurons[:, nw_i])

                        umatrix[i, j] = 0.5 * (dist1 + dist2)
        
        plt.imshow(umatrix, cmap="bone")
        plt.title("U-matrix")
        axis = plt.gca()
        axis.axis("off")
        axis.set_aspect('equal', 'datalim')
        plt.show(block=False)

    def plot_dendrogram(self):
        # Get linkage matrix
        Z = linkage(self.neurons.T, "ward")
        c, coph_dists = cophenet(Z, pdist(self.neurons.T))
        plt.figure()        
        dendrogram(Z)
        print("Plotted dendrogram with cophenetic distance of {:.2f}".format(c))
        plt.show(block=False)

    def get_closer(self, x):
        distances = np.linalg.norm(self.neurons - x, axis=0)
        return np.argmin(distances)

    def plot_closer(self, x, figure=None, patch=None):
        if figure:
            plt.figure(figure.number)
        else:
            plt.figure()
        axis = plt.gca()
        #axis.cla()

        distances = np.linalg.norm(self.neurons - x, axis=0)
        closest_idx = np.argmin(distances)
        close_i, close_j = np.unravel_index(closest_idx, (self.h, self.w))

        if not patch:
            patch = axis.add_patch(Rectangle((close_j, self.h - close_i - 1), 1, 1, alpha=0.4, color="r"))
                    
            # Add grid and square for each neuron
            for i in range(self.h - 1, -1, -1):
                for j in range(self.w):
                    axis.add_patch(Rectangle((j, i), 1, 1, fill=False))                
            
            # Adjust visuals and plot
            plt.title("Hit Map")
            axis.axis("off")
            plt.xlim([-0.1,self.w+0.1])
            plt.ylim([-0.1,self.h+0.1])
            axis.set_aspect('equal', 'datalim')
        else:
            patch.set_xy((close_j, self.h - close_i - 1))
        return patch

        #plt.show()




