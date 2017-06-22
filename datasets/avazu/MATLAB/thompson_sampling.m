function model = thompson_sampling(X, Y)
%
%
%

tic

%X = permute(X,[1 3 2]);

T = size(X, 1);   % number of training samples
d = size(X, 2);   % dimension of the item vectors
K = size(X, 3);   % num of item vectors per round

% Thompson sampling hyperparameters
R = 0.01;
epsilon = 1/log(T);
delta = 0.1;
v = R * sqrt(24 / epsilon * d * log(1 / delta));

% algorithm parameters
UM_inv = eye(d);
w_hat = zeros(d, 1);
f = zeros(d, 1);

% store results
model.regret = zeros(1, T);
model.cregret = zeros(1, T);            % cumulative regret
model.tregret = 0;                      % total regret

for t = 1:T 
 
    if 0 == mod(t - 1, 1000)
        fprintf('%d \n', t-1);  
    end

    full_context = squeeze(X(t,:,:));   % items at round t
    
    % sample w tilde    
    w_tilde = mvnrnd(w_hat, v^2*UM_inv);
    % compute payoffs
    %estimated_reward_array = full_context * w_hat;
    payoff_array = full_context' * w_tilde';
    % take the best action
    [~, indexMax] = max(payoff_array);
    x = squeeze(X(t,:,indexMax))';       % take it as a column vector      
    
    %for k = 1 : K
    %    tmpX = X(t, :, k);        
    %    payoff(k) = tmpX * w_tilde;
    %end    
    
    f = f + Y(t, indexMax) * x;
    UM_inv = UM_inv - UM_inv * (x * x') * UM_inv ...
        / (1 + x'* UM_inv * x);
    w_hat = UM_inv * f;
    
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w_hat = w_hat;
model.f = f;

fprintf('\n');
toc