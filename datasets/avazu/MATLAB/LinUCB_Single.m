 function model = LinUCB_Single(X, Y, users, alpha)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LinUCB-IND - optimized version
%
% (C) Shuai Li, DiSTA, University of Insubria
%     Email: ShuaiLi.SLi@gmail.com
%     Feb. 20th, 2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
d = size(X,2);   % dimensions of the vectors
T = size(X,1);   % number of training samples
K = size(X,3);   % num of vectors per step

numUsers = max(users);

UM_inv = zeros(d,d,numUsers);

for i=1:numUsers
    UM_inv(:,:,i) = eye(d);
end

bu = zeros(d,numUsers);
wU = zeros(numUsers,d);

model.regret = zeros(1,T);
model.cregret = zeros(1,T); %cumulative regret
model.tregret = 0; %total regret

p=zeros(K,1);

for i=1:T              
	%select the context vector
        for j=1:K
            p(j)=X(i,:,j)*wU(users(i),:)' + alpha * sqrt(X(i,:,j) * UM_inv(:,:,users(i)) * X(i,:,j)' * log(i+1));
        end    
        %indexMax = max(find(p==max(p)));
        [~, indexMax] = max(p);
        
        %update b's for user
        bu(:,users(i)) = bu(:,users(i)) + Y(i,indexMax)*X(i,:,indexMax)';
                
        %update UM_inv of the user 
        tmp = UM_inv(:,:,users(i)) * X(i,:,indexMax)';
        UM_inv(:,:,users(i)) = UM_inv(:,:,users(i)) - ( tmp * tmp')/(1+ X(i,:,indexMax)* tmp);
                
        %update w
        wU(users(i),:) = UM_inv(:,:,users(i))*bu(:,users(i));
        
        %update regret
        model.regret(i)=max(Y(i,:))-Y(i,indexMax);
end

model.cregret=cumsum(model.regret);
model.tregret=sum(model.regret);

%uncomment the following line to see the topology of the final graph
%spy(usersGraph)
toc

end
