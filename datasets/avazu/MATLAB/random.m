function cregret = random(T, K, Y)
% Random guessing
%   
regret = zeros(1, T);
for round=1:T
    t = K*(round-1)+1;
    action = randi(K);
    regret(round) = 1 - Y(t+action-1);
end
cregret = cumsum(regret);
