%% artificial data
T = 20000;
d = 10;
K = 5;
classes = 10;
numUsers = 10;
[X,Y,users,user_models] = artificial_data_generator(T,d,K,classes,numUsers); % T,d,k

%% thompson sampling
thompson_single = TS_one(X,Y);

%% thompson sampling for each user
thompson_multi = TS_single(X,Y,users);

%% linUCB single
alpha = 0.2;
linUCB_single = LinUCB_One(X,Y, alpha);

%% thompson cab
gamma = 0.1;
model = thompson_cab(X, Y, users, gamma);

%% Cab
p = 1;
minused = 1;
model2 = CAB1_woow_fastened(X, Y, users, 0.12, 0.20, minUsed, p);

%% random
cregret = random(T, K, Y);

%% plot 

% plotting the cregret vs time 
train=1:T;
hold on
plot(train,thompson_single.cregret,'g','DisplayName','Thompson Sampling single')
plot(train,thompson_multi.cregret,'b','DisplayName','Thompson Sampling multi')
plot(train,model.cregret,'r','DisplayName','Thompson CAB')
plot(train,cregret,'y','DisplayName','Random')
%plot(train,linUCB_single.cregret,'r','DisplayName','linUCB single')
plot(train,model2.cregret,'m','DisplayName','Cab')

title('Artificial Data')
xlabel('Time')
ylabel('Cumulative regret')
legend('show')
