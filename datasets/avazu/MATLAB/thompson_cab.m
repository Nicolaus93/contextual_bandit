function model = thompson_cab(X, Y, users, gamma, p, K)
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

T = size(X, 1)/K;           % number of training samples
d = size(X, 2);             % dimension of the item vectors
numUsers = max(users)+1;    % number or users (+1 since 0 is included)
used = zeros(1, numUsers);  % number of times each user gets served
minUsed = 1;

% Thompson sampling parameters
R = 0.01;
epsilon = 1/log(T);
delta = 0.1;
v = R * sqrt(24 / epsilon * d * log(1 / delta));

% users parameters
UM_inv = zeros(d, d, numUsers); % user inverse
f = zeros(d, numUsers);         % f
w_hat = f;                      % w^

for i = 1 : numUsers
    UM_inv(:,:,i) = eye(d);
end

% store results
model.regret = zeros(1, T);
model.cregret = zeros(1, T);            % cumulative regret
model.tregret = 0;                      % total regret
model.neighborhoodsize = zeros(1, T);   % tracks (selected) neighborhood size
model.updatedsize = zeros(1, T);        % tracks no. of updated weigth vectors w

for round = 1:T 
 
    if 0 == mod(round - 1, 100)
        fprintf('%d \n', round-1);
    end
    
    t = K*(round-1)+1;
    user = users(t)+1;                  % user at round t
    full_context = X(t:t+K-1,:);        % items at round t
    used(user) = used(user) + 1;        % update counter 
    
    % initialize neighboroods parameters
    N = zeros(K,numUsers);     % neigborhood sets   
    CB = zeros(K,numUsers);    % "confidence bounds for every user
    
    % store mu tilde
    mu_tilde = zeros(d, numUsers);
    
    % compute user parameters
    mu_tilde(:,user) = mvnrnd(w_hat(:,user), v^2*UM_inv(:,:,user));
    user_estimated_reward_array = full_context * w_hat(:,user);
    user_score_array = full_context * mu_tilde(:,user);
    CB(:,user) = abs(user_score_array - user_estimated_reward_array);
    
    % compute neighborhood sets (iterate over users since we can vectorize
    % over actions)
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
       end 
    end
    
    % computes associated aggregate quantities for the model estimate and
    % confidence bound    
    
    %for k = 1 : K
    %    mu_tilde_sum = sum(mu_tilde(:, find(N(k,:))),2);       
    %    mu_tilde_avg = mu_tilde_sum / sum(N(k,:));
    %    payoff(k) = full_context(k,:) * mu_tilde_avg;    
    %end
    
    % vectorized
    mu_tilde_sum = mu_tilde * N';
    mu_tilde_avg = bsxfun(@rdivide, mu_tilde_sum, sum(N,2)');
    payoff = sum(full_context .* mu_tilde_avg', 2);
    
    [~, indexMax] = max(payoff);
    x = X(t+indexMax-1, :)';
       
    % update parameters
    f(:, user) = f(:, user) + Y(t+indexMax-1) * x;
    UM_inv(:,:,user) = UM_inv(:,:,user) - UM_inv(:,:,user) * x * x' * UM_inv(:,:,user) ...
       / (1 + x'* UM_inv(:,:,user) * x);
    w_hat(:,user) = UM_inv(:,:,user) * f(:, user);
    if CB(indexMax,user) > gamma/4 * log(t+1)
        continue
    else
        to_update = find(N(indexMax,:));
        for j = 1:sum(N(indexMax,:))
            if CB(indexMax,to_update(j)) < gamma/4 * log(t+1)
                f(:, j) = f(:, j) + Y(t+indexMax-1) * x;
                UM_inv(:,:,j) = UM_inv(:,:,j) - UM_inv(:,:,j) * x * x' * UM_inv(:,:,j) ...
                    / (1 + x'* UM_inv(:,:,j) * x);
                w_hat(:,j) = UM_inv(:,:,j) * f(:, j);
            end
        end
    end          
    
    % update regret
    model.regret(round) = 1 - Y(t+indexMax-1);
    model.neighborhoodsize(round) = sum(N(indexMax,:));
    %model.updatedsize(t) = updated;
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w_hat = w_hat;
model.f = f;

fprintf('\n');
toc

%{
% plotinng the cregret vs time 
train=1:T;
%hold on
plot(train,model.cregret,'r','DisplayName','Thomp_CAB')

title('Avazu')
xlabel('Time')
ylabel('Cumulative regret')
%}

%legend('show')