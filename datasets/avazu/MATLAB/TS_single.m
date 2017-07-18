function model = TS_single(X, Y, users)
%
%
%

tic

T = size(X, 1);   % number of training samples
d = size(X, 2);   % dimension of the item vectors
K = size(X, 3);
numUsers = max(users);        

% Thompson sampling hyperparameters
R = 0.01;
epsilon = 1/log(T/numUsers);
delta = 0.1;
v = R * sqrt(24 / epsilon * d * log(1 / delta));

% users parameters
UM_inv = zeros(d, d, numUsers);     % user inverse
f = zeros(d, numUsers);             % f
w_hat = f;                          % w^
w_tilde = zeros(d, numUsers);       % w~

for i=1:numUsers
    UM_inv(:,:,i) = eye(d);
end

% store results
model.regret = zeros(1, T);
model.cregret = zeros(1, T);            % cumulative regret
model.tregret = 0;                      % total regret

for t = 1:T 
 
    if 0 == mod(t - 1, 10000)
        fprintf('%d \n', t-1);  
    end
    
    user = users(t);                    % user at round t
    full_context = squeeze(X(t,:,:));   % items at round t
            
    % sample w tilde    
    w_tilde(:,user) = mvnrnd(w_hat(:,user), v^2*UM_inv(:,:,user));
    % compute payoffs    
    payoff_array = w_tilde(:,user)' * full_context;
    % take the best action
    maxval = max(payoff_array);
    idx = find(payoff_array == maxval);
    indexMax =  idx(randi(numel(idx)));
    %[~, indexMax] = max(payoff_array);
    x = squeeze(X(t,:,indexMax))';       % take it as a column vector               
    
    % update user
    f(:, user) = f(:, user) + Y(t, indexMax) * x;
    UM_inv(:,:,user) = UM_inv(:,:,user) - UM_inv(:,:,user) * (x * x') * UM_inv(:,:,user) ...
        / (1 + x'* UM_inv(:,:,user) * x);
    w_hat(:,user) = UM_inv(:,:,user) * f(:, user);
        
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w_hat = w_hat;
model.f = f;

fprintf('\n');
toc
