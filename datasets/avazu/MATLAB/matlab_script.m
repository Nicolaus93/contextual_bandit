%% read datasets

X = csvread('filtered100_1M/processed.csv',1,1);
Y = csvread('filtered100_1M/reward_list.csv',1,1);
users = csvread('filtered100_1M/users.csv',1,1);
d = 28;
K = 10;
T = size(X, 1) / K;

%% thompson cab
gamma = 0.2;
p = 0.1;
model = thompson_cab(X, Y, users, gamma, p, K);

%% random
cregret = random(T, K, Y);

%% plot 

% plotinng the cregret vs time 
train=1:T;
hold on
plot(train,model.cregret,'r','DisplayName','Thomp_CAB')
plot(train,cregret,'b','DisplayName','Random')

title('Avazu')
xlabel('Time')
ylabel('Cumulative regret')