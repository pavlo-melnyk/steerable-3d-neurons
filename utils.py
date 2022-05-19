import torch
import torch.nn as nn
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
import itertools

from models import PointCMLP, SteerableModel



EPSILON = 1e-8



def construct_filter_banks(spheres, return_init_rotations=False, verbose=False):
    spheres = spheres.reshape(-1, 5) 
          
    # Step 1) compute the rotations R_O^k, i.e., from the original sphere centers to (1,1,1):
    original_centers = unembed_points(spheres)      # (n_spheres x 3)

    # normalize to compute the cross_product appropriately:
    norm_original_centers = original_centers / np.sqrt(np.sum(original_centers**2, axis=1, keepdims=True))
    
    # the angle between the original sphere center and (1,1,1):
    thetas = np.arccos(norm_original_centers @ np.sqrt([1/3, 1/3, 1/3]))

    # the corresponding rotation axis:
    n0s = np.array([np.cross(norm_original_centers[i], np.sqrt([1/3, 1/3, 1/3])) for i in range(len(norm_original_centers))])
    n0s = n0s / np.sqrt(np.sum(n0s**2, axis=1, keepdims=True))
    
    # construct the rotation matrices (from sphere centers to (1,1,1))
    rotations_0 = [rotation_matrix(n0s[i], thetas[i]) for i in range(len(thetas))]
    rotations_0_isom = [construct_rotation_isom(rotations_0[i]) for i in range(len(rotations_0))]
    
    # rotate the original spheres into (1,1,1)
    # (in R^5):
    rotated_original_spheres = [
        transform_points(
            spheres[i],
            construct_rotation_isom(rotations_0[i])
        ) 
        for i in range(len(spheres))
    ]
    
        
    # Step 2) compute the tetrahedron rotations R_{Ti}, i.e., rotations transforming (1, 1, 1) into the other three vertices:
    all_vertices = np.array([(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)])
    
    # the vertex-edge-vertex angle in a regular tetrahedron: 
    alpha = np.arccos(-1/3)
    
    # the rotation axes as the normalized cross products of the corresponding vertex pairs:
    cross_prods = np.array([np.cross(all_vertices[0], another_vertex) for another_vertex in all_vertices[1:]])
    rotation_axes = cross_prods / np.sqrt(np.sum(cross_prods**2, axis=1, keepdims=True))
    
    # obtain SO(3) from the axis-angle representation:
    tetra_rotations = [np.eye(3)] # to include the original spheres in the right poisitions
    tetra_rotations += [rotation_matrix(n, alpha) for n in rotation_axes]
    

    # Step 3) construct the filter banks B(S):
    # rotate *directly* in the conformal R^5 space (isomorphic to ME^3)
    # (already includes the INVERSE of the rotations from the original sphere centers to (1,1,1)):
    filter_banks = [
        transform_points(
            rotated_original_spheres[i],
            construct_rotation_isom(rotations_0[i].T @ tetra_rotation)
        ) 
        for i in range(len(spheres))
        for tetra_rotation in tetra_rotations
    ]
    
 
    if verbose:
        print('\noriginal centers:\n', original_centers)
        print('\nangles between the original sphere centers and (1,1,1):\n', thetas)
        print('\nrotated original spheres[0]:', rotated_original_spheres[0])
        print('\noriginal spheres[0]:', spheres[0])
        print('\nrotations_0[0]:', rotations_0[0])
        # print('\ntetra_rotations:\n', np.array(tetra_rotations))
        # print('\ntetra_rotation @ tetra_rotation.T:\n', np.array([tetra_rotation @ tetra_rotation.T for tetra_rotation in tetra_rotations]))
        # print('\ndet(tetra_rotations):\n', np.linalg.det(tetra_rotations))


    if return_init_rotations:
        return np.array(rotations_0), np.reshape(filter_banks, (-1, 5))    
    return np.reshape(filter_banks, (-1, 5))



def transform_parameters(model):
    ##### 1) extract the parameters (i.e., spheres) from the ancestor model:
    original_state_dict = model.state_dict()

    # get the geometric neuron spheres:
    hidden_name = 'hidden_layers.0.weight' 
    hidden_spheres = original_state_dict[hidden_name].detach()
    hidden_spheres_numpy = hidden_spheres.detach().cpu().numpy()

    # get the output layer spheres:
    out_name = 'out_layer.weight'
    out_spheres = original_state_dict[out_name].detach()
    out_spheres_numpy = out_spheres.cpu().numpy()
    
    
    ###### 2) construct filter banks:
    init_rotations, transformed_hidden_spheres = construct_filter_banks(hidden_spheres_numpy, return_init_rotations=True)
    
    # print(transformed_hidden_spheres.shape)
    
    # reshape to the right form:
    transformed_hidden_spheres = transformed_hidden_spheres.reshape(hidden_spheres.shape[0], -1)
    
    return init_rotations, hidden_spheres.data.new(transformed_hidden_spheres), out_spheres



def build_steerable_model(input_shape, output_dim, hidden_layer_sizes, init_axis_angle, transformed_parameters, print_hidden_layer_output=False):
    init_rotations, transformed_hidden_spheres, out_spheres = transformed_parameters
    # print('init_rotations.shape:', init_rotations.shape)
    # print('transformed_hidden_spheres.shape:', transformed_hidden_spheres.shape)
    
    ###### 3) instantiate the steerable model
    # the model has 3 free parameters -- the rotation to compute the interpolation coefficients;
    # now the ancestor model hidden layer parameters are transformed into filter banks, but the output (classification) layer
    # is kept the same:
    transformed_model = SteerableModel(init_axis_angle=init_axis_angle, 
                                       init_rotations=init_rotations, 
                                       print_hidden_layer_output=print_hidden_layer_output, 
                                       input_shape=input_shape, 
                                       output_dim=output_dim, 
                                       hidden_layer_sizes=hidden_layer_sizes, 
                                       bias=False)
     
    
    ###### 4) write in the parameters:   
    state_dict = transformed_model.state_dict()
    hidden_name = 'hidden_layers.0.weight' 
    # print('initial hidden spheres:', state_dict[hidden_name])
    state_dict[hidden_name].copy_(torch.autograd.Variable(transformed_hidden_spheres))

    # the output weights are the same as in the original:
    out_name = 'out_layer.weight'
    # print('initial output spheres:', state_dict[out_name])
    state_dict[out_name].copy_(torch.autograd.Variable(out_spheres))

    # write in:
    transformed_model.load_state_dict(state_dict)
    
    
    ##### 5) freeze all the parameters except the 3 rotation parameters:
    for name, param in transformed_model.named_parameters():
        if name != 'axis_angle':
            param.requires_grad = False
        
    return transformed_model



def entropy(x, is_logits=False):
    def _check_the_sum(x):
        x_sum = torch.sum(x, dim=-1)
        ones_like_sum = x_sum.new(torch.ones_like(x_sum))
        epsilon = x_sum.new(1e-6*torch.ones_like(x_sum))
        
        assert torch.all(torch.abs(x_sum - ones_like_sum) < epsilon), str((x_sum - ones_like_sum).item())

    if is_logits:
        x = torch.softmax(x, dim=-1)  
   
    _check_the_sum(x)

    return -torch.sum(x*torch.log(x+EPSILON))



def random_axis_angle(angle=None):
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    if angle is None:
        angle = np.pi * np.random.rand()
    return angle*axis



def chordal_distance(R1, R2):
    ''' Given two rotation matrices, 
    computes the angular difference between the two
    (the smallest).'''
    scaled_frob_norm = np.sqrt(np.sum((R1 - R2)**2) / 8)
    return 2*np.arcsin(scaled_frob_norm)



def torch_cross_operator(n):
    # this does not allow gradient computations:
    # return torch.as_tensor(
    #     [[0, -n[2], n[1]], 
    #      [n[2], 0, -n[0]],
    #      [-n[1], n[0], 0]]    
    # )

    # this does:
    N = torch.zeros(3,3)
    N[0,1] = -n[2]
    N[0,2] =  n[1]
    N[1,0] =  n[2]
    N[1,2] = -n[0]
    N[2,0] = -n[1]
    N[2,1] =  n[0]

    return N
    


def torch_rotation_matrix(axis_angle):
    return torch.matrix_exp(torch_cross_operator(axis_angle))



def identity(x):
    # needed in this format to save the model properly
    return x



def build_mlgp(input_shape=(4, 3), output_dim=8, hidden_layer_sizes=[4], bias=False, activation=identity):
    # Multilayer Geometric Perceptron
    print('\nmodel: MLGP')
    model = PointCMLP(input_shape, output_dim, hidden_layer_sizes, activation, bias, version=1)
    return model



def score(y, t):
    return torch.mean((torch.argmax(y, axis=1) == t).double()).item()



def save_checkpoint(state, save_dir='pretrained_models'):
    torch.save(state, save_dir+'/'+state['name']+'.tar')



def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))



def get_tetris_data():
    
    '''
    Inspired by
    https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/shape_classification.ipynb
    '''
    
    label_names = ['chiral_shape_1', 'chiral_shape_2', 'square', 'line', 'corner', 'L', 'T', 'zigzag']

    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # zigzag


    Xtrain = np.array([np.array(points_) for points_ in tetris])
    Ytrain = np.arange(len(Xtrain)) # [0, 1, ..., 7]

    return torch.from_numpy(Xtrain).float(), torch.from_numpy(Ytrain).long(), label_names
    


def construct_isomorphism_transformation(rotation, translation):
    '''
    Given 3D rotation and translation, constructs a matrix isomorphism of the transformation in R^{5} (the matrix itself is R^{5x5})
    corresponding to a motor in R^{3+1, 1} = ME^{3}.

    Args:
        rotation:    3D rotation, an array of shape (3, 3);
        translation: 3D translation, a vector of length 3;
    Returns:
        TR: a 5x5 matrix
    '''
    rotation_isom = construct_rotation_isom(rotation)
    translation_isom = construct_translation_isom(translation)

    TR = np.matmul(translation_isom, rotation_isom)
    
    return TR



def construct_rotation_isom(rotation):
    rotation = rotation.reshape(3, 3)
    bottom_part = np.zeros((2, 3))                                     
    rotation_isom = np.concatenate((rotation, bottom_part), axis=0)    

    right_part = np.eye(5)[:, -2:]                                     
    rotation_isom = np.concatenate((rotation_isom, right_part), axis=1)

    return rotation_isom 



def construct_translation_isom(translation):
    translation = translation.reshape(1, 3)
    base = np.eye(3)                              
                                                
    translation_isom = np.concatenate((base, translation, np.zeros((1, 3))), axis=0)    

    t_sq_mag = np.sum(translation**2, axis=-1, keepdims=True)                        
    right_part = np.concatenate((translation, 0.5*t_sq_mag, [[1.]]), axis=1)  

    translation_isom = np.concatenate((translation_isom, np.transpose([[0., 0., 0., 1., 0.]]), np.transpose(right_part)), axis=1)

    return translation_isom 



def embed_points(points):
    ''' 
    Performs conformal embedding -- embeds points in the conformal space.

    Args:
        points - the 3D model points, a tensor of shape (num_points, 3).
    Returns:
        embedded points - points embedded in R^{5}, a tensor of shape (num_points, 5).
    '''

    # compute the squared magnitude for each point:
    points_sq_mag = np.sum(points**2, axis=-1, keepdims=True)
    embedded_points = np.concatenate([points, 0.5*points_sq_mag, np.ones_like(points_sq_mag)], axis=-1)

    return embedded_points



def transform_points(points, transformation):
    ''' 
    Applies the isomorphism transformation to embedded points.

    Args:
        points:              points embedded in R^{5}, an array of shape (num_points, 5);
        transformation:      an array of shape (5, 5).
    Returns:
        transformed points:  a tensor of the same shape as the input points.
    '''

    # reshape to (1, 5, 5) to perform matmul properly:
    T = np.reshape(transformation, (-1, 5, 5))
    
    # expand dims to make a tensor of shape(num_points, 5, 1) to perform matmul properly:
    X = np.expand_dims(points, -1)

    # transform each point:
    transformed_points = np.matmul(T, X)
    
    # reshape to the input points size -- squeeze the last dimension:
    transformed_points = np.squeeze(transformed_points, -1)
        
    return transformed_points



def unembed_points(embedded_points):
    ''' 
    Performs a mapping that is inverse to conformal embedding.

    Args:
        embedded_points: points embedded in R^{5}, an array of shape (num_points, 5).
    Returns:
        points:          3D points, an array of shape (num_points, 3).

    '''

    # p-normalize points, i.e., divide by the last element:
    normalized_points = embedded_points / np.expand_dims(embedded_points[:,-1], axis=-1)

    # the first three elements are now Euclidean R^{3} coordinates:
    points = normalized_points[:,:3]

    return points



def plot_shapes(shapes, label_names, offset=0.1):
    ''' For visualizing the Tetris shapes.'''
    fig = plt.figure(1, figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, shape in enumerate(shapes):
        ax.scatter(shape[:, 0]+offset*i, shape[:, 1]+offset*i, shape[:, 2]+offset*i, s=90, label=label_names[i])
    ax.legend()



def get_2D_rotation(alpha):   
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array(((c, -s), (s, c)))



def derotate_points(points):
    '''
    For preprocessing the 3D skeleton data.
    Assuming the input size to be n_points x 3 (i.e., NxD)
    and the input is centered at the origin.
    '''
    assert len(points.shape) == 2
    
    # extract xy-coordinates of the points:
    points_xy = points[:,(0,2)] # N X 2
    # print(points_xy.shape)
    
    # hip_joints = (16, 12, 0)
    # define the "forward" vector using the core joint locations:
    v1 = points[12,:] - points[16,:]
    v2 = points[0,:] - points[16,:]
    v3 = np.cross(v1, v2)
    
    # define the orientation in the xy-plane:
    alpha = np.arctan2(v3[2], v3[0])
    
    # compute the de-rotation matrix:
    R = get_2D_rotation(-alpha)
        
    # de-rotate the xy_coordinates:
    points_xy_derot = points_xy @ R.T
    # print(points_xy_derot.shape)
    
    # assemble the coordinates:
    new_points = np.concatenate([points_xy_derot[:,0:1], points[:,1:2], points_xy_derot[:,1:2]], axis=1)
    # print(new_points.shape)
    
    return new_points, -alpha



def derotate_skeletons(skeletons, visualize=False, return_rotations=False):
    '''
    An unvectorized version.
    Assuming the input size to be n_shapes x n_points x 3 (i.e., BxNxD)
    and each shape is centered at the origin.
    
    If return_orig_rot is True, also return the rotation angle about y-axis for R_xz that brings the skeleton to its original pose, i.e., 
    skeleton = derotated_skeleton @ R_xz.T.
    '''
    derotated_skeletons = np.empty_like(skeletons)
    rotations = []
    for n in range(len(skeletons)):
        derotated_skeletons[n], alpha = derotate_points(skeletons[n])
        rotations.append(alpha)
    
    if visualize:
        visualize_skeleton(skeletons[n].T, get_edges(skeletons[n]))
        plt.title('centered skeleton')
        visualize_skeleton(derotated_skeletons[n].T, get_edges(derotated_skeletons[n]))
        plt.title('derotated skeleton') 
        
    assert np.all(skeletons.shape == derotated_skeletons.shape)
    
    if return_rotations:
        return derotated_skeletons, np.array(rotations)
    
    return derotated_skeletons



def center_points(points):
    '''
    Centers the input point clouds to the origin.
    Assuming the input size to be n_shapes x n_points x 3 (i.e., BxNxD).
    '''
    # center:
    mean_coords = np.mean(points, axis=1, keepdims=True)
    points -= mean_coords

    return points



def get_edges(points):
    '''
    Returns the skeleton edges (assuming a 20-joint skeleton) given a bag of 3D points.
    '''
    joint_pairs = [
        (0, 1), (1, 2), (2, 3), (2, 4), (2, 8),                                       # head and torso
        (8, 9), (9, 10), (10, 11), (4, 5), (5, 6), (6, 7),                            # arms
        (0, 16), (16, 17), (17, 18), (18, 19), (0, 12), (12, 13), (13, 14), (14, 15), # legs 
    ]
    
    return [points[joint_pair,:] for joint_pair in joint_pairs]



def visualize_skeleton(points, edges,  ax=None, show_grid=False, size=100, linewidth=5, is_utkinect=True, elev=10., azim=240., plot_forward_vector=True):
    fig = plt.figure()
    if ax is None:
        ax = fig.gca(projection='3d')
    ax.patch.set_alpha(0)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(show_grid)

    ax.scatter(points[0,:], points[2,:], points[1,:], c=points[2,:], cmap='plasma_r', s=size)

    # highlight the hip joints:
    hip_joints = (12, 16, 0)
    ax.scatter(points[0,hip_joints], points[2,hip_joints], points[1,hip_joints], c='b', s=size*2)

    if plot_forward_vector:
        v1 = points[:,12] - points[:,16]
        v2 = points[:,0] - points[:,16]
        v3 = np.cross(v1.flatten(), v2.flatten())
        ax.quiver(points[0,16], points[2,16], points[1,16],  v3[0], v3[2], v3[1], color='r', length=33, arrow_length_ratio=0.25)

    [ax.plot(edge[:,0], edge[:,2], edge[:,1], c='#472c7a', linewidth=linewidth) for edge in edges]

    set_axes_equal(ax)



def plot_confusion_matrix(cm, classes, normalize=False,  fontsize='small', title='Confusion matrix', cmap=plt.cm.Blues):

    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                 fontsize=fontsize,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def set_axes_equal(ax):    
    '''
    See 
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    by karlo [answered Jul 12 '15 at 4:13]
    
    Makes axes of a 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])