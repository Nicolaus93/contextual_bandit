%% preprocessing
features = csvread('../filtered100_1M/processed.csv',1,1);  % avoid reading column and row names
rewards = csvread('../filtered100_1M/reward_list.csv',1,1); 
users_id = csvread('../filtered100_1M/users.csv',1,1);

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
    users(t) = users_id(i)+1;       % we add 1 since 0 is included (it might cause problems with indices)
    t = t + 1;
end

%% artificial data
T = 10000;
d = 15;
k = 10;
classes = 5;
numUsers = 2;
[X,Y,users,user_models] = artificial_data_generator(T,d,k,classes,numUsers); % T,d,k

%% thompson sampling
thompson = thompson_sampling(X,Y);

%% thompson cab
gamma = 0.1;
p = 1;
minUsed = 1;
model = thompson_cab(X, Y, users, gamma, p, minUsed);

%% Cab
p = 0.5;
model2 = CAB1_woow_fastened(X, Y, users, 0.12, 0.20, minUsed, p);

%% random
cregret = random(T, k, Y);

%% vectorized thompson cab
addpath(genpath('/mtimesx_20110223/'));
cd 'mtimesx_20110223'
model3 = vect_thompson_cab(X, Y, users, gamma);

%% plot 

% plotting the cregret vs time 
train=1:T;
hold on
plot(train,thompson.cregret,'y','DisplayName','Thompson Sampling')
plot(train,model.cregret,'b','DisplayName','Thompson CAB')
plot(train,cregret,'r','DisplayName','Random')
plot(train,model2.cregret,'g','DisplayName','Cab')

title('Avazu')
xlabel('Time')
ylabel('Cumulative regret')
legend('show')
