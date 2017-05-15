%% preprocessing
features = csvread('filtered100_1M/processed.csv',1,1);  % avoid reading column and row names
rewards = csvread('filtered100_1M/reward_list.csv',1,1); 
users_id = csvread('filtered100_1M/users.csv',1,1);

K = 10;                         % items per round
T = size(features,1) / K;       % number of rounds
d = size(features,2);           % number of features

X = zeros(T,K,d);
Y = zeros(T,K);
users = zeros(T,1);
t = 1;
for i=1:K:T*K
    X(t,:,:) = features(i:i+9,:);
    Y(t, :) = rewards(i:i+9);
    users(t) = users_id(i);
end
