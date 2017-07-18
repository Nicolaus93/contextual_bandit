function model = thompson_cab2(X, Y, users, gamma, p, minUsed)
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

X = permute(X,[1 3 2]);           % use this to get the corresponding version of CAB

T = size(X, 1);                     % number of training samples
d = size(X, 3);                     % dimension of the item vectors
K = size(X, 2);                     % num of item vectors per round
% careful here
numUsers = max(users);            % number or users (+1 since 0 is included)
used = zeros(1, numUsers);          % number of times each user gets served

% Thompson sampling parameters
R = 0.01;
epsilon = 1/log(T);
delta = 0.2;
v = R * sqrt(24 / epsilon * d * log(1 / delta));

% users parameters
UM_inv = zeros(d, d, numUsers);     % user inverse
f = zeros(d, numUsers);             % f
w_hat = f;                          % w^

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
 
    if 0 == mod(t - 1, 5000)
        fprintf('%d \n', t-1);  
        avg = zeros(d, K, numUsers);
    end
    
    user = users(t);                    % user at round t
    full_context = squeeze(X(t,:,:));   % items at round t
    used(user) = used(user) + 1;        % update counter 
        
    % initialize neighboroods parameters
    N = zeros(K,numUsers);              % neigborhood sets   
    CB = zeros(K,numUsers);             % "confidence bounds" for every user
    
    % store mu tilde
    mu_tilde = zeros(d, numUsers);
    
    %if t == 1
    %    updated = zeros(1, numUsers);
    %end
        
    % sample mu tilde only if updated
    %to_sample = find(used);   
    %to_sample = find(used & updated);
    %for j = to_sample        
    %    mu_tilde(:,j) = mvnrnd(w_hat(:,j), v^2*UM_inv(:,:,j));            
    %end        
        
    % compute user parameters
    mu_tilde(:,user) = mvnrnd(w_hat(:,user), v^2*UM_inv(:,:,user));
    user_estimated_reward_array = full_context * w_hat(:,user);
    user_score_array = full_context * mu_tilde(:,user);
    CB(:,user) = abs(user_score_array - user_estimated_reward_array);
    %CB_user = CB(:,user);
    
    %{
    % compute neighborhood sets (iterate over users since we can vectorize
    % over actions)    
    s = repmat(mu_tilde(:,user),1,K);
    temp = find(used);    
    M = ones(1,10);
    N(:,user) = 1;
    ind = randperm(length(temp));
    temp = temp(ind);
    for j = temp
        if j == user 
           continue
        elseif rand < p
            mu_tilde(:,j) = mvnrnd(w_hat(:,j), v^2*UM_inv(:,:,j));
            estimated_reward_array = full_context * w_hat(:,j);
            score_array = full_context * mu_tilde(:,j);
            CB(:,j) = abs(score_array - estimated_reward_array);
            nearness = abs(estimated_reward_array - user_estimated_reward_array);
            threshold = CB(:,j) + CB(:,user);
            k = nearness<threshold;
            N(k,j) = 1;
            M(k) = M(k) + 1;            
            s(:,k) = bsxfun(@plus, s(:,k), mu_tilde(:,j));
            out = check_norm(s,M);
            if out               
                break
            end                       
        end
    end
    %}
            
    for j = 1 : numUsers
       if j == user           
           N(:,user) = 1;       % add user to neighbourood           
           continue  
       elseif used(j) > minUsed && rand < p
           mu_tilde(:,j) = mvnrnd(w_hat(:,j), v^2*UM_inv(:,:,j));
           estimated_reward_array = full_context * w_hat(:,j);
           score_array = full_context * mu_tilde(:,j);
           CB(:,j) = abs(score_array - estimated_reward_array);
           nearness = abs(estimated_reward_array - user_estimated_reward_array);
           threshold = CB(:,j) + CB(:,user);
           k = nearness<threshold;
           N(k,j) = 1; 
           if 0 == mod(t - 1, 5000)
                avg(:,:,j) = bsxfun(@rdivide, mu_tilde * N', sum(N,2)');
           end
       end 
    end    
    
    %{
    a = zeros(K,numUsers);
    for i = 1:K
        for j = 1:numUsers
            a(K,j) = norm(squeeze(avg(:,K,j)));
        end
    end
    %}
        
    
    % computes associated aggregate quantities for the model estimate and
    % confidence bound            
    mu_tilde_sum = mu_tilde * N';
    mu_tilde_avg = bsxfun(@rdivide, mu_tilde_sum, sum(N,2)');
    payoff = sum(full_context .* mu_tilde_avg', 2);    
    % best action and its index
    [~, indexMax] = max(payoff);
    x = squeeze(X(t,indexMax,:));     % take it as a column vector      
        
    updated = zeros(1,numUsers);
    
    % update parameters    
    if CB(indexMax,user) > gamma/4 
        f(:, user) = f(:, user) + Y(t, indexMax) * x;
        UM_inv(:,:,user) = UM_inv(:,:,user) - UM_inv(:,:,user) * (x * x') * UM_inv(:,:,user) ...
            / (1 + x'* UM_inv(:,:,user) * x);
        w_hat(:,user) = UM_inv(:,:,user) * f(:, user);
        updated(user) = user;
    else
        to_update = find(N(indexMax,:));
        % check parfor
        for j = to_update
            if CB(indexMax, j) < gamma/4
                f(:, j) = f(:, j) + Y(t, indexMax) * x;
                UM_inv(:,:,j) = UM_inv(:,:,j) - UM_inv(:,:,j) * (x * x') * UM_inv(:,:,j) ...
                    / (1 + x'* UM_inv(:,:,j) * x);
                w_hat(:,j) = UM_inv(:,:,j) * f(:, j);
                updated(j) = j;
            end
        end                
    end
    
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);
    model.neighborhoodsize(t) = sum(N(indexMax,:));
    model.updatedsize(t) = sum(find(updated));
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w_hat = w_hat;
model.f = f;

fprintf('\n');
toc