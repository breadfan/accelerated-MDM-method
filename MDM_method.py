import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def GenerPoints(a, alpha, num_points, dim):
    return alpha * np.random.rand(num_points, dim) + a

def GetHullandPlot(points, dim):
    hull = ConvexHull(points)
    if dim == 2:
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'b-')
    elif dim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.plot(points[:, 0], points[:, 1], points[:, 2], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'b-')
    else:
        print('There is no way to plot graph in dim > 3, but hull has found successfully!')
    return hull


class MDM(object):
    __class__ = 'MDM'
    __doc__ = """
        This is an implementation the accelerated Mitchell-Demyanov-Malozemov method for 
        finding nearest to coordinates beginning point.
        Also plots convex-hull and optimal solution in 2- and 3-dimensional cases.
    """

    def __init__(self, points, hull, dim, accel):
        self._dim = dim
        self._points = points.copy()
        self._hull = hull
        self._A_matrix = points.copy().transpose()
        self.isAccelerated = accel                  #which method we're using
        self.iterations = None
        self.delta_p = None
        self.p_vector = None
        self.vector_current = None
        self.supp_vector = None                     #supp for vector p (i.e. {i \in 0 : dim - 1 | p[i] > 0} )


    def solve(self):
        V = 0
        iterations = 0

        delta_p = 1
        p_vector = [0 for i in range(0, len(self._points))]
        supp_vector = []
        t_param_vector =[]

        MIN_set = []
        MAX_set = []
        diff_vector = []                            #for cycles finding
        P_vectors = []                              #matrix of p_vectors
        V_vectors = []
        cycle_constructed = False
        cycle_is_constructing = False
        special_upd_done = False                    #whether special update Wn = W + lambdaV is done
        cycle_current_size = 0                      #we will search actual size of cycle

        initial_approximation = 1                    #it can be changed for lowering iterations sake;
        #for first approximation we'll just take point from a board of hull - cause it's easy reduced
        vector_current = self._points[self._hull.vertices[initial_approximation]].copy()    #need copy() there for non-changing _points
        supp_vector.append(self._hull.vertices[initial_approximation])                      #approximation => get vect_0
        p_vector[self._hull.vertices[initial_approximation]] = 1                            #working right concat
        #then we need to find vect_{k+1} iteratively

        while delta_p > 0.000001 and iterations < 500 and len(supp_vector) != 0:
            if self.isAccelerated is True and cycle_constructed is True and special_upd_done is False:
                for i in range(cycle_size):  #constructing V as linear combination of D's that we used previously
                    V += -1 * t_param_vector[cycle_start + i] * diff_vector[cycle_start + i]
                p_vector = P_vectors[cycle_start]                   #returning to value where cycle had begun
                vector_current = V_vectors[cycle_start]       #returning
                supp_vector = []                        #returning
                for i in range(len(p_vector)):          #returning
                    if p_vector[i] > 0.0000001:
                        supp_vector.append(i)

                lambda_t = -np.dot(vector_current, V) / np.linalg.norm(V) ** 2
                for i in range(cycle_size):
                    if t_param_vector[i] > 0:
                        if lambda_t > (1 - p_vector[MIN_set[i]]) / t_param_vector[i]:
                            lambda_t = (1 - p_vector[MIN_set[i]]) / t_param_vector[i]
                    elif t_param_vector[i] < 0:
                        if lambda_t > -p_vector[MAX_set[i]] / t_param_vector[i]:
                            lambda_t = -p_vector[MAX_set[i]] / t_param_vector[i]
                vector_current += lambda_t * V
                for i in range(cycle_size):
                    p_vector[MAX_set[i]] -= lambda_t * t_param_vector[i]
                    p_vector[MIN_set[i]] += lambda_t * t_param_vector[i]
                special_upd_done = True     #once it's done we're forgiving about that


            mult = np.dot(self._points[supp_vector], vector_current)
            ind_max = np.argmax(mult)           #finding max for indices in supp_vector
            ind_max = supp_vector[ind_max]      #finding max general in our mult product

            mult = np.matmul(vector_current, self._A_matrix)
            ind_min = np.argmin(mult)                                                 # i''_k
            if self.isAccelerated is True and cycle_constructed is False:
                MIN_set.append(ind_min)
                MAX_set.append(ind_max)
            diff = self._points[ind_max] - self._points[ind_min]
            print('\nDifference: ' + str(diff))
            delta_p = np.dot(diff, vector_current)

            if delta_p > 0.000001:                  #if not bigger, then we've found a solution
                print('delta_p: ' + str(delta_p))
                print('p_vector[ind_max] = ' + str(p_vector[ind_max]) + '\nnp.linalg.norm(diff)): '
                      + str(np.linalg.norm(diff)))
                t_param = delta_p /(p_vector[ind_max] * (np.linalg.norm(diff)) ** 2)  # recounting all variables
                if t_param >= 1:
                    t_param = 1

                if self.isAccelerated is True:          #if using accelerated MDM-method
                    if iterations > 0 and cycle_is_constructing is False:  #constructing cycle(active finding cycle, i mean, active-active)
                        contains = np.where(np.all(diff_vector == diff, axis = 1))[0]    #finds if diff_vector contains diff
                        if len(contains) != 0:      #found first element of cycle
                            cycle_is_constructing = True        #cycle is constructing now
                            cycle_start = contains[0]                     #index of first element of cycle; not changing
                            cycle_size = iterations - cycle_start         #not changing
                            cycle_current_size += 1         #this var for checking if all variables actually are cycle
                        P_vectors.append(p_vector.copy())
                        V_vectors.append(vector_current.copy())
                        t_param_vector.append(t_param)      #saving t_params for constructing V in the future
                        diff_vector.append(diff)            #saving D_i
                    elif cycle_is_constructing is True and cycle_constructed is False:
                        if cycle_current_size < cycle_size and \
                                np.where(np.all(diff_vector == diff, axis = 1))[0] \
                                == (cycle_start + cycle_current_size):
                            cycle_current_size += 1
                            diff_vector.append(diff)
                            t_param_vector.append(t_param)
                        else:
                            cycle_constructed = True
                            print('CYCLE FOUND AND CONSTRUCTED SUCCESSFULLY!')
                    elif iterations == 0:
                        P_vectors.append(p_vector.copy())
                        V_vectors.append(vector_current.copy())
                        t_param_vector.append(t_param)
                        diff_vector.append(diff)


                vector_current -= t_param * p_vector[ind_max] * diff
                supp_vector = []                #recounting
                temp1 = t_param * p_vector[ind_max]
                temp2 = (1 - t_param)
                p_vector[ind_min] += temp1
                p_vector[ind_max] *= temp2

                for i in range(len(p_vector)):
                    if p_vector[i] > 0.0000001:
                        supp_vector.append(i)
            print('Vector current: ' + str(vector_current))
            iterations += 1
            print('Iterations: ' + str(iterations))
            print('Supp_vector: ' + str(supp_vector))

        return vector_current





points =np.array([[ -73.337555  ,   -4.82192605],
       [   9.36299101,   14.79378288],
       [  33.74875017,   10.02043701],
       [ 133.04981839,   92.18760616],
       [-105.00396348,  -69.46640213],
       [  32.54560694,   43.96449265],
       [ -78.01174375,   61.08025333],
       [  92.03366094,  -51.6208306 ],
       [  17.22114877,   54.92524147],
       [ -87.14266467,  128.5875058 ],
       [ -35.76597696, -161.63324815],
       [ 156.36709765,  -55.60266369],
       [  41.00897625,  -54.92133061],
       [ 129.50005618,  -39.14660553],
       [ 101.99767049,    5.91893179],
       [ 120.62635591,   39.32842524],
       [  58.91037616,  -29.52086718],
       [-116.99548555,  -35.64041842],
       [ -49.26778003,   18.11377985],
       [  91.22017504,   26.95527778],
       [   5.98350205,  -29.65544224],
       [  73.8606758 ,  -67.33527561],
       [ -57.11269196,  -23.38066312],
       [  10.29413585,   19.91249178],
       [ -76.57980277,   36.15112039],
       [  40.91217006,  -17.81387299],
       [  51.88700332,  -69.65988091],
       [  57.41048001, -119.28130887],
       [ -66.49323658,  -92.43371661],
       [  10.46455101,  -80.23934518]])

dim = 2; number_of_points = 30

isManualEnter = False
isAccelerated = True

inp = input('Use manual enter or use default parameters? M/D.')
if inp == 'M':
    isManualEnter = True
if isManualEnter is True:
    gener = input('Use generator or manual input of points? G/M')
    if gener == 'M':
        print('Enter the data (points values) in the \"data.txt\" file.')
        points = []
        number_of_points = 0
        with open("data.txt") as f:
            for line in f:
                temp = [float(x) for x in line.split()]
                points.append(temp)
                number_of_points += 1
        points = np.array(points)
        dim = len(points[0])
    elif gener == 'G':
        dim = int(input('Enter number of dimensions: '))
        number_of_points = int(input('Enter number of points: '))
        points = GenerPoints(3, 68, number_of_points, dim)

    temp = input('Use classic or accelerated MDM-method? C/A')      #by default we're using accelerated method
    if temp == 'C':
        isAccelerated = False

elif isManualEnter is False:
    print('Our DEFAULT values: \nNumber of dimensions is ' \
          + str(dim) + '\nNumber of points is ' + str(number_of_points))


hull = GetHullandPlot(points, dim)
mdm = MDM(points, hull, dim, isAccelerated)
result = mdm.solve()                                    #returns a point in R^dim

if dim == 2 :
    plt.plot([result[0], 0], [result[1], 0], 'ro')
elif dim == 3 :
    plt.plot([result[0], 0], [result[1], 0], [result[2], 0], 'ro')
plt.show()
print('Result is: ' + str(result))