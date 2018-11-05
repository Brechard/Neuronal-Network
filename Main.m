clear, clc
%sigmoid

X=[ [1 0 0 0 0 0 0 0]
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
nodes_in_1 = n_nodes_in;
nodes_hid_2 = 3;
nodes_out_3 = n_nodes_in;

alpha = 0.8;
lambda = 0.0001;
std_dev = 0.01;

W_1 = normrnd(0, std_dev, nodes_hid_2, nodes_in_1);
b_1 = normrnd(0, std_dev, nodes_hid_2, 1);

W_2 = normrnd(0, std_dev, nodes_out_3, nodes_hid_2);
b_2 = normrnd(0, std_dev, nodes_out_3, 1);

D_W_1 = zeros(nodes_hid_2, nodes_in_1);
D_b_1 = zeros(nodes_hid_2, 1);

D_W_2 = zeros(nodes_out_3, nodes_hid_2);
D_b_2 = zeros(nodes_out_3, 1);

c = 0;
maxError = 1;

tic;
i = 0;
while abs(maxError) > 0.04
    i = i + 1;
    D_W_1 = zeros(nodes_hid_2, nodes_in_1);
    D_b_1 = zeros(nodes_hid_2, 1);

    D_W_2 = zeros(nodes_out_3, nodes_hid_2);
    D_b_2 = zeros(nodes_out_3, 1);
    output = X;
    maxError = 0;
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
        
        if(max(abs(delta_3)) > maxError)
            maxError = max(abs(delta_3));
        end
        errors(c) = max(delta_3);
    end
    
    % Code to print the weights, only works for assignment dimensions
%     for j = 0:7
%         w_1_values(i,1 + 3* j) = (W_1(1, j + 1));
%         w_1_values(i,2 + 3* j) = (W_1(2, j + 1));
%         w_1_values(i,3 + 3* j) = (W_1(3, j + 1));            
%     end
%     for k = 0:7
%         w_2_values(i,1 + 8* k) = (W_2(k + 1, 1));
%         w_2_values(i,2 + 8* k) = (W_2(k + 1, 2));
%         w_2_values(i,3 + 8* k) = (W_2(k + 1, 3));
%     end


% Update the parameters
    W_1 = W_1 - alpha * ((D_W_1 / n_samples) + lambda * W_1);
    b_1 = b_1 - alpha * ((D_b_1 / n_samples));

    W_2 = W_2 - alpha * ((D_W_2 / n_samples) + lambda * W_2);
    b_2 = b_2 - alpha * ((D_b_2 / n_samples));
end

time = toc;
% subplot(211);
% plot(w_1_values)
% title("Weights of layer 1");
% subplot(212);
% plot(w_2_values)
% title("Weights of layer 2");
plot(errors);