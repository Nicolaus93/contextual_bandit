function model = CAB1_woow_fastened(X, Y, users, alpha, gamma, minused, p)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% EHE: This is a basic context-aware clustering procedure
% EHE: recomputes from scratch neighborhood sets
% EHE: minused is the minimum no. of times a user has to be served in order
%     to be a candidate for inclusion in a neighborhood set
% EHE: In this procedure the aggregate neighborhoods are computed using the
%       CAB1 ('Average Aggregate') Algorithm
% CG: Added on Dec 24th: parameter p is a "fastener" probability. Each user
% j gets added to the neighborhood of users(t) only with probability p
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%X = permute(X,[1 3 2]);

T = size(X, 1);   % number of training samples
d = size(X, 2);   % dimension of the item vectors
K = size(X, 3);   % num of item vectors per round

tic

numUsers = max(users);              % number of users
used = zeros(1, numUsers);          % no of times each user gets served

% user parameters
UM_inv = zeros(d, d, numUsers);     % for user inverse
bu = zeros(d, numUsers);            % for user
wU = bu;                            % for user   

for i = 1 : numUsers
    UM_inv(:,:,i) = eye(d);
end

model.regret = zeros(1, T);
model.cregret = zeros(1, T);            % cumulative regret
model.tregret = 0;                      % total regret
model.neighborhoodsize = zeros(1, T);   % tracks (selected) neighborhood size
model.updatedsize = zeros(1, T);        % tracks no. of updated weigth vectors w

for t = 1 : T 

    if 0 == mod(t - 1, 100)
        fprintf('%d ', t-1);
    end

    % counter
    used(users(t)) = used(users(t)) + 1;
    
    N = zeros(K, numUsers);   % neigborhood sets
    Nc = zeros(K,1);          % cardinality of neighorhood sets 
    CB = zeros(numUsers,K);
    
   	% computes neighborhood sets
    Z = rand(K,numUsers); % generates, for each instance vector, the subset of users to be considered for inclusion 
    for k = 1 : K
        tmpX = X(t, :, k);        
        for j = 1 : numUsers
            if ((Z(k,j) <= p) && (used(j) >= minused)) || (j == users(t))  % CG: added on Dec 23rd in order to speed up computation
                CB(j,k) = alpha *sqrt(tmpX * UM_inv(:, :, j) * tmpX')*(log(t + 1));
            end
            if (((Z(k,j) <= p) && (used(j) >= minused)) || (j == users(t))) && (abs(tmpX * (wU(:,users(t)) - wU(:,j)) <= CB(users(t),k) + CB(j,k)));
               N(k,Nc(k)+1) = j;
               Nc(k) = Nc(k)+1;
           end
        end
    end
    
    % computes associated aggregate quantities for the model estimate and
    % confidence bound
    payoff = zeros(K, 1);
    for k = 1 : K
        tmpX = X(t, :, k);
        sumw = zeros(d,1);
        CB_ag = 0;
        for j = 1 : Nc(k)
            sumw = sumw + wU(:,N(k,j));
            CB_ag = CB_ag + CB(N(k,j),k);
        end
        avgw = sumw/Nc(k);
        avgCB = CB_ag/Nc(k);     % average of the aggregate confidence bound
        payoff(k) = tmpX * avgw + avgCB ;
    end
    
    [~, indexMax] = max(payoff);
    tmpXmax = X(t, :, indexMax); 
    %%%%%
    updated = 0;  
    if CB(users(t),indexMax) > gamma/4
        bu(:, users(t)) = bu(:, users(t)) + Y(t, indexMax) * tmpXmax';
        tmp = UM_inv(:, :, users(t)) * tmpXmax' ;
        UM_inv(:, :, users(t)) = UM_inv(:, :, users(t)) - (tmp * tmp') / (1 + tmpXmax * tmp);
        wU(:,users(t)) = UM_inv(:, :, users(t)) * bu(:, users(t));
        updated = 1;
    else
        for j = 1 : Nc(indexMax)
           if CB(N(indexMax,j),indexMax) <= gamma/4
              bu(:, N(indexMax,j)) = bu(:, N(indexMax,j)) + Y(t, indexMax) * tmpXmax';
              tmp = UM_inv(:, :, N(indexMax,j)) * tmpXmax' ;
              UM_inv(:, :, N(indexMax,j)) = UM_inv(:, :, N(indexMax,j)) - (tmp * tmp') / (1 + tmpXmax * tmp);
              wU(:,N(indexMax,j)) = UM_inv(:, :, N(indexMax,j)) * bu(:, N(indexMax,j));
              updated = updated + 1;
            end
        end
    end
   
    % update regret
    model.regret(t) = max(Y(t, :)) - Y(t, indexMax);
    model.neighborhoodsize(t) = Nc(indexMax);
    model.updatedsize(t) = updated;
end

model.cregret = cumsum(model.regret);
model.tregret = sum(model.regret);
model.w = wU;
model.b = bu;

fprintf('\n');
toc

% plotinng the cregret vs time 
train=1:T;
%hold on
%plot(x,y1,'DisplayName','sin(x)')
plot(train,model.cregret,'r','DisplayName','CAB1')

title('Artificial Data')
xlabel('Time')
ylabel('Cummulative regret')

%legend('show')

end


