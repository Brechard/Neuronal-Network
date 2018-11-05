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

%h_Wb=f(W'.*X);
[n_nodes_in, n_samples] = size(X);

% Number of nodes in each layer
nodes_in_1 = 8;
nodes_hid_2 = 3;
nodes_out_3 = 8;

alpha = 0.01;
lambda = 0.001;

W_1 = normrnd(0, 0.01, nodes_hid_2, nodes_in_1);
b_1 = normrnd(0, 0.01, nodes_hid_2, 1);

D_W_1 = zeros(nodes_hid_2, nodes_in_1);
D_b_1 = zeros(nodes_hid_2, 1);

W_2 = normrnd(0, 0.01, nodes_out_3, nodes_hid_2);
b_2 = normrnd(0, 0.01, nodes_out_3, 1);

D_W_2 = zeros(nodes_out_3, nodes_hid_2);
D_b_2 = zeros(nodes_out_3, 1);

c = 0;

for i = 1:1000
    for sample = 1:n_samples
        c = c + 1;
        a_1 = X(:, sample);
        
        z_2 = W_1 * X(:, sample) + b_1;
        a_2 = 1 ./ (1+exp(-(z_2)));
        
        z_3 = W_2 * a_2 + b_2;
        a_3 = 1 ./ (1+exp(-(z_3)));
        
        delta_3 = -(Y(:, sample) - a_3) .* (a_3 .* (1 - a_3));
        
        delta_2 = (W_2' * delta_3) .* (a_2 .* (1  - a_2));
        
        J_W_1 = delta_2 * a_1';
        J_b_1 = delta_2;
        
        J_W_2 = delta_3 * a_2';
        J_b_2 = delta_3;
        
        D_W_1 = D_W_1 + J_W_1;
        D_b_1 = D_b_1 + J_b_1;
        
        D_W_2 = D_W_2 + J_W_2;
        D_b_2 = D_b_2 + J_b_2;

        errors(c) = std(delta_3);
    end
    
    W_1 = W_1 - alpha * ((D_W_1 / n_samples) + lambda * W_1);
    b_1 = b_1 - alpha * ((D_b_1 / n_samples));
    
    W_2 = W_2 - alpha * ((D_W_2 / n_samples) + lambda * W_2);
    b_2 = b_2 - alpha * ((D_b_2 / n_samples));
end
plot(errors);