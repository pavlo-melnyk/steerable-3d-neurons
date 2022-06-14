import torch
import torch.nn as nn
import numpy as np



class SteerableModel(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_layer_sizes=[], 
                 activation=lambda x: x, bias=False, 
                 init_axis_angle=torch.ones(4), init_rotations=None,
                 print_hidden_layer_output=False):
        
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.input_shape = input_shape
        self.n_input_points = input_shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes        
        self.f = activation      
        self.hidden_layer_activations = []
        self.print_hidden_layer_output = print_hidden_layer_output      
            
        # Step 0. Create model layers (the same as for the ancestor MLGP) 
        # hidden layers:
        hidden_layers = []
       
        # for input_shape = (k, 3), e.g., 3D shape coordinates, 
        # each row of the input sample (R^3) is embedded in R^5 (ME^3);
        # the resulting (k x 5)-array is vectorized row-wise and fed 
        # to the first layer;
        # each subsequent hidden layer output R^n is first embedded in R^(n+2)
        # and then fed to the next layer:
        M1 = 4 * input_shape[0] * (input_shape[1] + 2) # the "4" is for the number of the basis functions in 3D
        for M2 in hidden_layer_sizes:
            layer = nn.Linear(M1, M2, bias=bias)
            hidden_layers.append(layer)
            M1 = M2 + 2

        self.hidden_layers = nn.ModuleList(hidden_layers)

        # the output layer:
        try:
            self.out_layer = nn.Linear(M2 + 2, output_dim, bias=bias)   
        except UnboundLocalError:
            self.out_layer = nn.Linear(M1, output_dim, bias=bias)   
                      
                
        # Step 1.              
        
        # Initial interpolation coefficients v^k
        # that will be transformed according to each geometric neuron sphere originally learned:
        self.v = 0.5 * torch.ones(4).to(self.device)  # (the number of the interpolation coefficients for the 3D case is 4) 
        
        # Initial rotations for geometric neuron spheres learned originally:
        # init_rotations[k] is the rotation R_O^k from the originally learned k-th sphere center to (1,1,1) 
        if init_rotations is not None:            
            # init_rotations.shape = (n_input_points*n_geometric_neurons x 3 x 3)
            self.init_rotations = torch.from_numpy(init_rotations).to(self.device).float()
        
        # The orthogonal basis matrix M:
        self.M = 0.5 * torch.from_numpy(
                            np.array(
                                [[ 1,  1,  1, 1], 
                                 [ 1, -1, -1, 1], 
                                 [-1,  1, -1, 1], 
                                 [-1, -1,  1, 1]]
                            ).T
                        ).to(self.device).float()
        
        # We want to compute the interpolation coefficients v^k
        # that depend on the rotation applied to the input, 
        # For the proof-of-concept experiments, we assume that the parameters of this rotation,
        # i.e., self.axis_angle parameters, are known.
        # In the forward pass, we will construct a rotation matrix from the axis-angle representation
        # and apply it "sandwiched" with the respective original sphere initial rotations R_O^k
        # to the interpolation coefficients v^k, and multiply it by the basis matrix M.
        
        # Initialize the axis-angle representation parameters:
        self.axis_angle = nn.Parameter(
            torch.as_tensor(init_axis_angle).to(self.device)
        ) 
        

                
    def forward(self, x):
        v = self.v # a constant 4-vector, i.e., 0.5*torch.ones(4)
        
        # Construct a 3x3 rotation matrix from the axis-angle (free) parameters:
        R_B = torch_rotation_matrix(self.axis_angle).float().to(self.device)        
        
        
        # Step 2. Transform v for each sphere with the corresponding intitial rotation and the obtained rotation R_B (possibly, regressed)
        # as v[:3] = R_O^k @ R_B @ R_O^k.T @ v[:3], and then multiply by the basis matrix self.M (equation 15 in the paper):
        vs = [] 
        for i in range(len(self.init_rotations)): # self.hidden_layer_sizes[0]) * self.n_input_points rotations
            transformation = self.init_rotations[i] @ R_B @ self.init_rotations[i].T
            rotated_v =  transformation @ v[:3]
            rotated_v = torch.cat([rotated_v, v[-1:]]) @ self.M 
            vs.append(rotated_v)
        vs = torch.cat(vs, dim=-1)

        # repeat for an element-wise scaling done further:
        vs = vs.repeat_interleave(5)
            
        # reshape to (n_geometric_neurons x n_input_points x 4 x 5):
        vs = vs.reshape(self.hidden_layer_sizes[0], self.n_input_points*len(v)*5)

    
    
        # Step 3. Perform the scaling of each coordinate of the input point by the corresponding interpolation coefficient
        # (one way to implement the steerability constraint (16) in the paper):
        
        # repeat the input n_geometric_neuron_times:
        x = x.repeat(1, self.hidden_layer_sizes[0], 1)
        
        # reshape into (batch_size, n_geometric_neurons, n_selected_points, 3):
        x = x.reshape(len(x), self.hidden_layer_sizes[0], self.n_input_points, x.shape[-1])
     
        # quadruplicated each point since we now have 4*n_original_spheres per geometric neuron:
        x = x.repeat_interleave(4, dim=2)        # (batch_size x n_geometric_neurons x 4*n_input_points x 3)

        # the same embedding as for the MLGP:
        embed_term_1 = -torch.ones(len(x), x.shape[1], x.shape[2], 1).to(self.device).float()
        embed_term_2 = -torch.sum(x**2, axis=-1) / 2 

        x = torch.cat((x, embed_term_1, embed_term_2.view(-1, x.shape[1], x.shape[2], 1)), dim=-1)
        # (batch_size x n_geometric_neurons x 4*n_input_points x 5)
        
        # reshape to (batch_size x n_geometric_neurons x 4*n_input_points*5):
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])   
        
        # scale the input point X^k with V^k:
        # (scaling each coordinate of the input point and then taking the dot product with the filter bank is the same
        #  as scaling the dot product)             
        x = x * vs
    
        # reshape to (batch_size x n_geometric_neurons x 1 x 4*n_input_points*5):
        x = x.unsqueeze(-2)

                

        # Step 4. Forward-propagate:
        
        is_first_hidden_layer = True
        for layer in self.hidden_layers:
            if is_first_hidden_layer:               
                # the filter banks:
                filter_banks = layer.weight # filter_banks.shape = (n_geometric_neurons x 4*n_input_points*5)
                
                # reshape to (1 x n_geometric_neurons x 4*n_input_points*5 x 1):
                filter_banks = filter_banks.unsqueeze(0).unsqueeze(-1)
                
                # (batch_size x n_geometric_neurons x 4*n_input_points*5) 
                # @ (1 x n_geometric_neurons x 4*n_input_points*5 x 1) =
                # = (batch_size x n_geometric_neurons x 1 x 1)
                
                x = torch.matmul(x, filter_banks)

                # squeeze the last two dimensions:                
                x = x[:,:,0,0] # (batch_size x n_geometric_neurons)

                # store the activations:
                self.hidden_layer_activations = x.detach()        
  
                x = self.f(x)
    
                if self.print_hidden_layer_output:
                     print('\nhidden_layer_output:\n', x)
                
            else:
                x = self.f(layer(x))

            embed_term_1 = -torch.ones(len(x), 1).to(self.device).float()
            embed_term_2 = -torch.sum(x**2, axis=1).view(-1, 1) / 2

            x = torch.cat((x, embed_term_1, embed_term_2), dim=1)
            
            is_first_hidden_layer = False
        
        x = self.out_layer(x)

        return x




class PointCMLP(nn.Module):   
    def __init__(self, input_shape, output_dim, hidden_layer_sizes=[], activation=lambda x: x, bias=False, version=1):
        ''' 
        Args: 
            input_shape:        a list/tuple of 2 integers; the size of one input sample, i.e., (n_rows, n_columns);
                                the model input is, however, expected to be a 3D array of shape (n_samples, n_rows, n_columns);
            hidden_layer_sizes: a list of integers containing the number of hidden units in each layer;
            activation:         activation function, e.g., nn.functional.relu;
            output_dim:         integer; the number of output units.
            version:            either 0 or 1: 
                                0 to create a vanilla MLP (the input will be vectorized in the forward function);
                                1 to create the MLGP or the MLHP.
                                For the former, the embedding of the input is row-wise.
                                In order to create the latter, one needs to vectorize each sample in the input, i.e.,
                                reshape the input to (n_samples, 1, n_rows*n_columns).
        '''
        super().__init__()

        self.input_shape = input_shape
        self.f = activation
        self.hidden_layer_activations = [] # to store the first hidden layer activations for the ancestor MLGP
        
        # create hidden layers:
        hidden_layers = []

        if version == 0:
            # for vanilla MLP:
            M1 = np.prod(input_shape)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(layer)
                M1 = M2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            # the output layer:
            try:
                self.out_layer = nn.Linear(M2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)                   

            self.forward = self.forward_0


        elif version == 1:
            # for the MLGP and the MLHP

            # for input_shape = (k, 3), e.g., 3D shape coordinates, 
            # each row of the input sample (R^3) is embedded in R^5 (ME^3);
            # the resulting (k x 5)-array is vectorized row-wise and fed 
            # to the first layer;
            # each subsequent hidden layer output R^n is first embedded in R^(n+2)
            # and then fed to the next layer

            M1 = input_shape[0] * (input_shape[1] + 2)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(layer)
                M1 = M2 + 2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            #  the output layer:
            try:
                self.out_layer = nn.Linear(M2 + 2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)   

            self.forward = self.forward_1
                        
        
    def forward_0(self, x):
        # for the vanilla MLP

        # vectorize each sample:
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        for layer in self.hidden_layers:
            x = self.f(layer(x))
        x = self.out_layer(x)

        return x


    def forward_1(self, x): 
        # for the MLGP and the MLHP   

        embed_term_1 = -torch.ones(len(x), x.shape[1], 1).float()
        embed_term_2 = -torch.sum(x**2, axis=2) / 2 

        if torch.cuda.is_available():
            embed_term_1 = embed_term_1.to(x.device)

        x = torch.cat((x, embed_term_1, embed_term_2.view(-1, x.shape[1], 1)), dim=2)
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        is_first_hidden_layer = True
        for layer in self.hidden_layers:
            x = layer(x)

            if is_first_hidden_layer:
                # store the activations:
                self.hidden_layer_activations = x.detach() 

            x = self.f(x)

            embed_term_1 = -torch.ones(len(x), 1).float()
            embed_term_2 = -torch.sum(x**2, axis=1).view(-1, 1) / 2

            if torch.cuda.is_available():
                embed_term_1 = embed_term_1.to(x.device)

            x = torch.cat((x, embed_term_1, embed_term_2), dim=1)

            is_first_hidden_layer = False

        x = self.out_layer(x)

        return x




def torch_cross_operator(n):   
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
