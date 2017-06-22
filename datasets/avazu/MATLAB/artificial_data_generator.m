function [X,Y,users,model] = artificial_data_generator(T,d,K,classes,numUsers)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CG: generates artificial data by randomly perturbing linear functions
% T = time horizon
% d = dimension of the instance vectors
% K = no. of instance vectors per round
% n = no. of distict users
%
% X =     T x d x K dimensional
% Y =     T x K dimensional
% users = T x 1 dimensional
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
% generate users features
model1 = [1,0,1,0,0,0,1,0,1,1];
model2 = [1,0,0,1,0,0,1,0,1,1];
model3 = [0,1,0,1,0,1,0,1,0,1];
model4 = [0,1,1,0,1,1,0,1,0,0];
model5 = [0,0,0,0,1,1,0,0,1,1];

users1 = zeros(10,K);
users2 = zeros(10,K);
users3 = zeros(10,K);
users4 = zeros(10,K);
users5 = zeros(10,K);

for i = 1:10
    a = rand(1,K)/10;
    users1(i,:) = (model1 + a) / sum(model1 + a);
    a = rand(1,K)/10;
    users2(i,:) = (model2 + a) / sum(model2 + a);
    a = rand(1,K)/10;
    users3(i,:) = (model3 + a) / sum(model3 + a);
    a = rand(1,K)/10;
    users4(i,:) = (model4 + a) / sum(model4 + a);
    a = rand(1,K)/10;
    users5(i,:) = (model5 + a) / sum(model5 + a);    
end

user_feat = [users1; users2; users3; users4; users5];
%}

user_feat = zeros(classes*numUsers,d);

% check if all 0s!
model = randi([0,1],classes,d);

for i = 1:classes
    for j = 1:numUsers
        a = rand(1,d)/10;
        user_feat(j*i,:) = (model(i,:)+a) / sum(model(i,:)+a);   
    end
end

% generates users
users = randi(size(user_feat,1), T, 1);

% generates contexts
X = zeros(T,d,K);
for t = 1:T
    Xt = randi([0,1],K,d);
    %Xt = eye(K,d);
    X(t,:,:) = bsxfun(@rdivide,Xt,sqrt(sum(Xt.^2,2)))';
    if find(isnan(X(t,:,:)))        
        X(t,isnan(X(t,:,:))) = zeros(size(X(t,isnan(X(t,:,:)))));
        %X(t,isnan(X(t,:,:))) = [1,0,0,0,0,0,0,0,0,0];
    end
end

% generates Y
Y = zeros(T,K);
for t = 1:T
    mean = user_feat(users(t),:)*squeeze(X(t,:,:));
    %Y(t,:) = mvnrnd(mean, 0.1 * eye(K));
    for i = 1:K
        Y(t,i) = normrnd(mean(i),0.1);        
    end
end

end
        