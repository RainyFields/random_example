%% generate possible states and init values at time t

function [dyn_states,dyn_values,len] = gen_dyn_states(t)
len = (t+1)*(t^2/2 - (t*(2*t+1)/6)+(t+1));
len = int16(len);
dyn_states = zeros(len,4);
dyn_values = zeros(len,1);
% n1 = 0...t, w1 = 0...n1; n2 = t-n1, w2 = 0....t-n1
i = 0;
for n1 = 0:t
    for w1 = 0:n1
        for w2 = 0:t-n1
            i = i+1;
            n2 = t-n1;
            dyn_states(i,:) = [n1,w1,n2,w2];
        end
    end
end

end

