%% Bayesian solution of the two arm bandit problem through dynamic
% programming
clear all; clf; close all;

% parameter setting
pa = 0.4;
pb = 0.6;
EPOCH = 100;
h = 20; % number of pulls


% recording all states and values
tic
for t = h:-1:1
    if t == h
        [states_t,values_t,lens_t] = gen_dyn_states(t);
        hist_s = {states_t};
        hist_v = {values_t};
    end
    if t~=1
        [prev_states, prev_v, prev_len] = gen_dyn_states(t-1);
        hist_s(h-t+2) = {prev_states};
        for i = 1:prev_len
            % agent take action 1 at time t-1
            rho1 = (prev_states(i,2)+1)/(prev_states(i,1)+2);
            state_r = prev_states(i,:)+[1,1,0,0];
            state_nr = prev_states(i,:)+[1,0,0,0];
            next_state = hist_s(h-t+1);
            next_state = next_state{1,1};
            r_idx = find(ismember(next_state,state_r,'rows'));
            nr_idx = find(ismember(next_state,state_nr,'rows'));
            v_r = hist_v(h-t+1);
            v_r = v_r{1,1};
            v_r = v_r(r_idx);
            v_nr = hist_v(h-t+1);
            v_nr = v_nr{1,1};
            v_nr = v_nr(nr_idx);
            v1 = rho1 + rho1*v_r+(1-rho1)*v_nr;
            
            % agent take action 2 at time t-1
            rho2 = (prev_states(i,4)+1)/(prev_states(i,3)+2);
            state_r = prev_states(i,:)+[0,0,1,1];
            state_nr = prev_states(i,:)+[0,0,1,0];
            next_state = hist_s(h-t+1);
            next_state = next_state{1,1};
            r_idx = find(ismember(next_state,state_r,'rows'));
            nr_idx = find(ismember(next_state,state_nr,'rows'));
            v_r = hist_v(h-t+1);
            v_r = v_r{1,1};
            v_r = v_r(r_idx);
            v_nr = hist_v(h-t+1);
            v_nr = v_nr{1,1};
            v_nr = v_nr(nr_idx);
            v2 = rho2+rho2*v_r+(1-rho2)*v_nr;
            if v1>v2
                v = v1;
            else
                v = v2;
            
            end
            prev_v(i) = v;
        end
        hist_v(h-t+2) = {prev_v};
    end
end


% forward pass to find the optimal strategy based on future expected reward (take pa pb into consideration)

rewards = zeros(h,EPOCH);
actions = zeros(h,EPOCH);
for epoch = 1:EPOCH
current_state = [0,0,0,0];
    for t = 1:h
    % selection actions
    if t==1
        % random choice
        a = rand;
        if a < 0.5
            action = 0;
        else
            action = 1;
        end
    else
        % pick according to the future expected reward
        rho1 = (current_state(2)+1)/(current_state(1)+2);
        s1 = current_state + [1,1,0,0];
        s2 = current_state + [1,0,0,0];
        states = hist_s(h-t+1);
        states = states{1,1};
        s1_idx = find(ismember(states,s1,'rows'));
        s2_idx = find(ismember(states,s2,'rows'));
        vs = hist_v(h-t+1);
        vs = vs{1,1};
        v1 = vs(s1_idx);
        v2 = vs(s2_idx);
        v_rho1 = rho1+rho1*v1+(1-rho1)*v2;
        
        rho2 = (current_state(4)+1)/(current_state(3)+2);
        s3 = current_state + [0,0,1,1];
        s4 = current_state + [0,0,1,0];
        s3_idx = find(ismember(states,s3,'rows'));
        s4_idx = find(ismember(states,s4,'rows'));
        vs = hist_v(h-t+1);
        vs = vs{1,1};
        v3 = vs(s3_idx);
        v4 = vs(s4_idx);
        v_rho2 = rho2+rho2*v3+(1-rho2)*v4;
        
        if v_rho1 > v_rho2
            action = 0;
        else
            action = 1;
        end
    end
    
    % get reward
    p = rand;
    if (action == 0 && p < pa) 
        reward = 1;
        temp = [1,1,0,0];
    elseif (action==1 && p<pb)
        reward = 1;
        temp = [0,0,1,1];
    elseif (action==0 && p>=pa)
        reward = 0;
        temp = [1,0,0,0];
    else
        reward = 0;
        temp = [0,0,1,0];
    end
    rewards(t,epoch) = reward;
    actions(t,epoch) = action;
    % update current state
    
    current_state = current_state+temp;
end
end
toc

%% analysis
% with different pa and pb and corresponding action rations
prob_a = length(find(rewards==0))/(EPOCH*h);
prob_b = length(find(rewards==1))/(EPOCH*h);
X = categorical({'prob left action','prob right action'});
Y = [prob_a,pa;prob_b,pb];
figure(2)
bar(X,Y)
title("probability of choosing left/right action")

% average rewards:
avg_rewards = sum(rewards);
figure(1)
plot(avg_rewards)
ylim([0,h])
title('culmulative rewards for 100 epochs')
