%% read data
[T,K] = size(Y);

%% random
cregret = random(T, K, Y);

%% thompson sampling
thompson_one= TS_one(X,Y);

%% linUCB
alpha = 0.2;
linUCB_one = LinUCB_One(X,Y, alpha);

%% linUCB for each user
alpha = 0.2;
linUCB_multi = LinUCB_Single(X, Y, users, alpha);

%% thompson sampling for each user
thompson_single= TS_single(X,Y,U);

%% thompson cab
gamma = 0.1;
model = thompson_cab(X, Y, U, gamma);

%% Cab
p = 0.5;
cab = CAB1_woow_fastened(X, Y, users, 0.12, 0.20, minUsed, p);

%% plot 

% plotting the cregret vs time 
train=1:T;
hold on
plot(train,thompson_one.cregret,'g','DisplayName','Thompson Sampling one')
plot(train,thompson_single.cregret,'b','DisplayName','Thompson Sampling multi')
plot(train,model.cregret,'r','DisplayName','Thompson CAB')
plot(train,cregret,'y','DisplayName','Random')
%plot(train,linUCB_multi.cregret,'m','DisplayName','linUCB multi')
%plot(train,linUCB_one.cregret,'r','DisplayName','linUCB one')
%plot(train,cab.cregret,'m','DisplayName','Cab')

title('Last FM')
xlabel('Time')
ylabel('Cumulative regret')
legend('show')
