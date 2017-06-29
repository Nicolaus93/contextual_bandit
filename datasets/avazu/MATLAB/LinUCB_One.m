function model = LinUCB_One(X, Y, alpha, avazu)
% To do
%
%

tic

if avazu
    X = permute(X,[1 3 2]); % use for Avazu
end
    
T = size(X, 1);
d = size(X, 2);

% Initialization
b = zeros(d, 1);
M_inv = eye(d);

model.regret = zeros(1, T);
model.cregret = zeros(1, T);    % cumulative
model.tregret = 0;              % total

for t = 1 : T
    
    if 0 == mod(t - 1, 10000)
        fprintf('%d \n', t-1);  
    end
    
    full_context = squeeze(X(t,:,:));   % items at round t    
    w = M_inv * b;   
    payoff_array = w' * full_context + alpha * (sqrt(diag(full_context' * M_inv * full_context) * log(t+1)))';
    
    % take the best action
    [~, indexMax] = max(payoff_array);    
    x = full_context(:,indexMax);
    
    % update parameters
    b = b + Y(t, indexMax) * x;
    M_inv = M_inv - M_inv * (x * x') * M_inv ...
        / (1 + x'* M_inv * x);    
    
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);

end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
toc
end
