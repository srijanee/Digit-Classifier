clear; clc

data = csvread('train.csv',1,0);


X = data(:,2:end);
Y = data(:,1);

for i=1:size(Y)
	if Y(i)==0
		Y(i)=10;
	end
end

input_layers = 784;
output_layers = 10;
hidden_layer_size = 400;

Theta1 = initialize(input_layers,hidden_layer_size);
Theta2 = initialize(hidden_layer_size,output_layers);

lambda = 1;

initial_param = [Theta1(:);Theta2(:)];


options = optimset('MaxIter', 50);

lambda = 1;

%costFunction = @(p) nnCost(p,...
%							input_layers,...
%							hidden_layer_size,...
%							output_layers, X, Y, lambda);

[nn_params, cost] = fmincg( @(p) (nnCost(p,...
							input_layers,...
							hidden_layer_size,...
							output_layers, X, Y, lambda)),initial_param, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layers + 1)), ...
                 hidden_layer_size, (input_layers + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layers + 1))):end),...
                 output_layers, (hidden_layer_size + 1));


pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);

test = csvread('test2.csv',0,0);
X_test = test(:,2:end);
Y_test=test(:,1);

for i=1:size(Y_test)
	if Y_test(i)==0
		Y_test(i)=10;
	end
end

pred = predict(Theta1,Theta2,X_test);
%sum(pred==Y_test)
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == Y_test)) * 100);

