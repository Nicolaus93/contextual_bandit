function model = thompson_cab(X, Y, users, gamma)
% Context aware clustering of bandits using Thompson Sampling
% Inputs:
%       - X
%       - Y
%       - users
%       - gamma 
%       - p
%       - K
% Output:
%       - model
tic

T = size(X, 1);                     % number of training samples
d = size(X, 2);                     % dimension of the item vectors
K = size(X, 3);                     % num of item vectors per round

% careful here
numUsers = max(users);              % number or users (+1 since 0 is included)
used = zeros(1, numUsers);          % number of times each user gets served

% Thompson sampling parameters
R = 0.01;
epsilon = 1/log(T/numUsers);
delta = 0.2;
v = R * sqrt(24 / epsilon * d * log(1 / delta));

% users parameters
UM_inv = zeros(d, d, numUsers);     % user inverse
f = zeros(d, numUsers);             % f
w_hat = f;                          % w^
w_tilde = zeros(d, numUsers);       % w~
updated = zeros(1, numUsers);

for i = 1 : numUsers
    UM_inv(:,:,i) = eye(d);
end

% store results
model.regret = zeros(1, T);
model.cregret = zeros(1, T);            % cumulative regret
model.tregret = 0;                      % total regret
model.neighborhoodsize = zeros(1, T);   % tracks (selected) neighborhood size
model.updatedsize = zeros(1, T);        % tracks no. of updated weigth vectors w

for t = 1:T 
 
    if 0 == mod(t - 1, 100)
        fprintf('%d \n', t-1);  
        %avg = zeros(d, K, numUsers);
    end
    
    user = users(t);                    % user at round t
    full_context = squeeze(X(t,:,:));   % items at round t
    used(user) = used(user) + 1;        % update counter 
        
    % initialize neighboroods parameters
    N = zeros(numUsers, K);    
    updated(user) = 1;
        
    % sample mu tilde only if updated
    to_sample = find(updated);
    for j = to_sample        
        w_tilde(:,j) = mvnrnd(w_hat(:,j), v^2*UM_inv(:,:,j));            
    end        
        
    estimated_reward = w_hat' * (full_context);
    payoff = w_tilde' * (full_context);
    CB = abs( payoff - estimated_reward);    
    N(bsxfun(@minus, estimated_reward, estimated_reward(user,:)) < ...
        bsxfun(@plus, CB, CB(user,:))) = 1;
    N(used==0, :) = 0;    
    
    % computes associated aggregate quantities for the model estimate and
    % confidence bound 
    avg_w_tilde = bsxfun(@rdivide, w_tilde * N, sum(N,1));
    payoff = sum(full_context .* avg_w_tilde,1);
    % best action and its index
    [~, indexMax] = max(payoff);
    x = full_context(:,indexMax);    
    
    % update parameters    
    updated = zeros(1,numUsers);
    % update user
    f(:, user) = f(:, user) + Y(t, indexMax) * x;
    UM_inv(:,:,user) = UM_inv(:,:,user) - UM_inv(:,:,user) * (x * x') * UM_inv(:,:,user) ...
        / (1 + x'* UM_inv(:,:,user) * x);
    w_hat(:,user) = UM_inv(:,:,user) * f(:, user);
    updated(user) = 1;
    % update neighborood
    if CB(user, indexMax) < gamma/4
        to_update = find(N(:,indexMax));        
        for idx = 1:numel(to_update)
            j = to_update(idx);                
            if CB(j,indexMax) < gamma/4
                f(:, j) = f(:, j) + Y(t, indexMax) * x;
                UM_inv(:,:,j) = UM_inv(:,:,j) - UM_inv(:,:,j) * (x * x') * UM_inv(:,:,j) ...
                    / (1 + x'* UM_inv(:,:,j) * x);
                w_hat(:,j) = UM_inv(:,:,j) * f(:, j);
                updated(j) = 1;
            end
        end                
    end
    
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);
    model.neighborhoodsize(t) = sum(N(:,indexMax));
    model.updatedsize(t) = sum(find(updated));
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w_hat = w_hat;
model.f = f;

fprintf('\n');
toc