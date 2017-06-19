function cregret = random(T, K, Y)
% Random guessing
%   
regret = zeros(1, T);
for t=1:T    
    action = randi(K);
    regret(t) = 1 - Y(t, action);
end
cregret = cumsum(regret);
