function cregret = random(T, K, Y)
% Random guessing
%   
regret = zeros(1, T);
for t=1:T    
    action = randi(K);     
    regret(t) = max(Y(t, :)) - Y(t, action);
end
cregret = cumsum(regret);
