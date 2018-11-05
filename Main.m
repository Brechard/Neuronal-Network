clear, clc
%sigmoid

X=[[1 0 0 0 0 0 0 0]
    [0 1 0 0 0 0 0 0]
    [0 0 1 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 1 0 0 0]
    [0 0 0 0 0 1 0 0]
    [0 0 0 0 0 0 1 0]
    [0 0 0 0 0 0 0 1]];

Y = X;
output = X;

%h_Wb=f(W'.*X);
[n_nodes_in, n_samples] = size(X);

% Number of nodes in each layer
nodes_in_1 = 8;
nodes_hid_2 = 3;
nodes_out_3 = 8;

alpha = 0.9;
lambda = 0.0001;

W_1 = normrnd(0, 0.01, nodes_hid_2, nodes_in_1);
b_1 = normrnd(0, 0.01, nodes_hid_2, 1);

W_2 = normrnd(0, 0.01, nodes_out_3, nodes_hid_2);
b_2 = normrnd(0, 0.01, nodes_out_3, 1);

D_W_1 = zeros(nodes_hid_2, nodes_in_1);
D_b_1 = zeros(nodes_hid_2, 1);

D_W_2 = zeros(nodes_out_3, nodes_hid_2);
D_b_2 = zeros(nodes_out_3, 1);

c = 0;
delta_3 = 1;
while max(abs(delta_3)) > 0.05
    D_W_1 = zeros(nodes_hid_2, nodes_in_1);
    D_b_1 = zeros(nodes_hid_2, 1);

    D_W_2 = zeros(nodes_out_3, nodes_hid_2);
    D_b_2 = zeros(nodes_out_3, 1);
    output = X;
    for sample = 1:n_samples
        c = c + 1;
% Perform the backpropagation algorithm
    % Perform a feedfoward pass
        % The activation function of the first layer is the inputs
        a_1 = X(:, sample);
        
        z_2 = W_1 * a_1 + b_1;
        a_2 = 1 ./ (1+exp(-(z_2)));
        
        z_3 = W_2 * a_2 + b_2;
        a_3 = 1 ./ (1+exp(-(z_3)));
        
        output(:, sample) = a_3;
    % Error of the last layer
        delta_3 = -(Y(:, sample) - a_3);

    % Error of layer 2 (hidden layer)
        delta_2 = (W_2' * delta_3) .* (a_2 .* (1  - a_2));
        
    % Calculate the partial derivatives
        J_W_1 = delta_2 * a_1';
        J_b_1 = delta_2;
        
        J_W_2 = delta_3 * a_2';
        J_b_2 = delta_3;
        
% Update the weights and bias matrix
        D_W_1 = D_W_1 + J_W_1;
        D_b_1 = D_b_1 + J_b_1;
        
        D_W_2 = D_W_2 + J_W_2;
        D_b_2 = D_b_2 + J_b_2;

        errors(c) = max(delta_3);

    end
% Update the parameters
    W_1 = W_1 - alpha * ((D_W_1 / n_samples) + lambda * W_1);
    b_1 = b_1 - alpha * ((D_b_1 / n_samples));

    W_2 = W_2 - alpha * ((D_W_2 / n_samples) + lambda * W_2);
    b_2 = b_2 - alpha * ((D_b_2 / n_samples));
end
plot(errors);