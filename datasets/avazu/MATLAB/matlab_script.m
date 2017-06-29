%% preprocessing

features = csvread('../filtered_20yes_100no/processed.csv',1,0);  % avoid reading column names
rewards = csvread('../filtered_20yes_100no/reward_list.csv',1,0); 
users_id = csvread('../filtered_20yes_100no/users.csv',1,0);

K = 10;                         % items per round
T = size(features,1) / K;       % number of rounds
%T = 10000;
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

%% random
cregret = random(T, K, Y);

%% thompson sampling
thompson_single = TS_one(X,Y);

%% thompson sampling for each user
thompson_multi = TS_single(X,Y,users);

%% linUCB single
alpha = 0.2;
linUCB_single = LinUCB_One(X,Y, alpha);

%% thompson cab
gamma = 0.1;
p = 1;
minUsed = 1;
model = thompson_cab(X, Y, users, gamma, p, minUsed);

%% Cab
p = 0.5;
model2 = CAB1_woow_fastened(X, Y, users, 0.12, 0.20, minUsed, p);

%% plot 

% plotting the cregret vs time 
train=1:T;
hold on
plot(train,thompson_single.cregret,'g','DisplayName','Thompson Sampling single')
plot(train,thompson_multi.cregret,'b','DisplayName','Thompson Sampling multi')
%plot(train,model.cregret,'b','DisplayName','Thompson CAB')
plot(train,cregret,'y','DisplayName','Random')
plot(train,linUCB_single.cregret,'r','DisplayName','linUCB single')
%plot(train,model2.cregret,'m','DisplayName','Cab')

title('Avazu')
xlabel('Time')
ylabel('Cumulative regret')
legend('show')
