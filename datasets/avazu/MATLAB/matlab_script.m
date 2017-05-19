%% preprocessing
features = csvread('../filtered100_1M/processed.csv',1,1);  % avoid reading column and row names
rewards = csvread('../filtered100_1M/reward_list.csv',1,1); 
users_id = csvread('../filtered100_1M/users.csv',1,1);

K = 10;                         % items per round
T = size(features,1) / K;       % number of rounds
d = size(features,2);           % number of features

T = 5000;
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

%% thompson cab
gamma = 0.2;
p = 0.6;
minUsed = 1;
model = thompson_cab(X, Y, users, gamma, p, minUsed);

%% random
cregret = random(T, K, Y);

%% Cab
model2 = CAB1_woow_fastened(X, Y, users, 0.12, 0.20, 1, 0.6);

%% plot 

% plotting the cregret vs time 
train=1:T;
hold on
plot(train,model.cregret,'r','DisplayName','Thompson CAB')
plot(train,cregret,'b','DisplayName','Random')
plot(train,model2.cregret,'y','DisplayName','Cab')

title('Avazu')
xlabel('Time')
ylabel('Cumulative regret')
legend('show')
