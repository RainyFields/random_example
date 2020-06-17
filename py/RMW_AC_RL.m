clear all; clf; close all
format LONG
% hyper parameters
d = 2; % diameters of the swimming pool
d_pf = 0.1; % diameters of the platform
v = 0.5; % swimming speed
delta_t = 0.1; % discrete time step
T = 120; % cutoff time
N = 493; % # of place cells
SIGMA = 0.16; 
p_inits = [1,0;-1,0;0,1;0,-1];
p_init = p_inits(randi([1,4]),:); % starting point
% platform center location
pf_c = rand([1,2]) * 2 - 1;

GAMMA = .85;
lr_actor = 0.003;
lr_critic = 0.007;

% place cell receptive field center
temp1 = rand(N,1);
temp2 = rand(N,1)*pi*2;
x = sqrt(temp1).*cos(temp2);
y = sqrt(temp1).*sin(temp2);
s = [x,y];
% x = linspace(-1,1,25);
% [s1,s2] = meshgrid(x);
% s = [s1(:),s2(:)];
% figure(13)
% plot(s(:,1),s(:,2),"*")

w = zeros([1,N])+eps; % critic weights
z = zeros([8,N])+eps; % actor weights

actions = [-v*delta_t,0;... 
    0,v*delta_t;... 
    v*delta_t,0;...
    0,-v*delta_t;...
    v*delta_t/sqrt(2),v*delta_t/sqrt(2);...
    v*delta_t/sqrt(2),-v*delta_t/sqrt(2);...
    -v*delta_t/sqrt(2),v*delta_t/sqrt(2);...
    -v*delta_t/sqrt(2),-v*delta_t*sqrt(2)];

TRIALS=23

% learning

for trial = 1:TRIALS
    trial
    i = 1;
current_p = p_init;
current_t = 0;
p_hist = zeros([T/delta_t,2]);
c_hist = zeros([T/delta_t,1]);
r_hist = zeros([T/delta_t,1]);
delta_hist = zeros([T/delta_t,1]);

while distance(current_p,pf_c)>d_pf && current_t<120
    
    current_t = current_t + delta_t;
    current_p;
    p_hist(i,:) = current_p;
    i = i+1;
    % actor-critic network
    f = exp(-(distance(current_p,s).^2)/(2*SIGMA^2));
    c_hist(i) = w*f;
    a = z*f;
    P = exp(2*a)/(sum(exp(2*a)));
    P(isnan(P)) = 1;
    cum_P = cumsum(P);
    temp = rand();
    done = false;
    idx = 1;
    while done == false
        if temp<=cum_P(idx)
            done = true;
            action_idx = idx;
        elseif idx<8
            idx = idx+1;
        else break 
        end
        
        
    end
    action = actions(action_idx,:);
    old_p = current_p;
    current_p = old_p + action;
    % bouncing conditions
    while distance(current_p,[0,0])>d/2
        action_idx = randi(8);
        action = actions(action_idx,:);
        current_p = old_p + action;
    end
    
    if distance(current_p,pf_c)>d_pf/2
        r_hist(i) = 0;
    else
        r_hist(i) = 1;
    end

    
    % learning
    g = zeros([1,8]);
    g(action_idx) = 1;
    delta_hist(i-1) = r_hist(i-1) + GAMMA*c_hist(i)-c_hist(i-1);
    delta_w = delta_hist(i-1).*f;
    w = w - lr_critic*delta_w';
    delta_z = delta_hist(i-1)*f*g;
    z = z - lr_critic*delta_z';
    
    
    
    
end

if trial == 1 || trial ==7|| trial ==4 ||trial==22
    if i<=1200
        p_hist_plot = p_hist(1:i-1,:);
    else
        p_hist_plot = p_hist;
    end
    figure(trial);
    
    subplot(2,1,1)
    hold on
    plot(p_hist_plot(:,1),p_hist_plot(:,2))
    
    text(p_init(1),p_init(2),'starting point')
    
    ezplot(@(x,y) (x-pf_c(1)).^2 + (y-pf_c(2)).^2 -d_pf^2)
    title("water maze reference memory")
    axis equal
    
    xlim([-1,1]);
    ylim([-1,1]);
    
    
    % plot c(p)
    c_p = zeros(N,1);
    
    subplot(2,1,2)
    
    for i = 1:N
        f = exp(-(distance(s(i,:),s).^2)/(2*SIGMA^2));
        c_p(i) = w*f;
    end
    
    xq = linspace(min(s(:,1)), max (s(:,1)));
    yq = linspace(min(s(:,2)), max (s(:,2)));
    [X,Y] = meshgrid(xq,yq);
    Z = griddata(s(:,1),s(:,2),c_p, X, Y, 'cubic');
    
    surf(X,Y,Z);
    % grid on
    
    
    
end
end    



%% testing part
% figure(1000)
% hold on
% ezplot(@(x,y) (x-0).^2 + (y-0).^2 -1^2)
% axis equal
% xlim([-1,1])
% ylim([-1,1])
% 
% scatter(0.781248916810278, 0.618492424049176,10)
% hold off
% 

