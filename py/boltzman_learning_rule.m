% boltzman machine learning rule
% N = 10 and generate binary binary patterns
% use generated data to compute the clamped statistics (x_i, x_j)_c, and
% (x_j)_c.
% learning steps K = 200, in each learning step use T = 500 steps of
% sequential stochastic dynamics to compute the free statistics (x_i, x_j)
% plotting the size of the change in weights versus iteration to test
% convergence

clc; close all; clear;

N = 10;
P = 100; % # training patterns
K = 200; %  learning steps
T = 500; % time steps
eta = 0.05; % learning rate

% dataset generation
s = 2 * randi([0,1], P, N)-1; 

% initialization
w = 2 * rand(N,N) - 1;
w = w - tril(w, -1) + triu(w, 1);
% w = w - tril(w, 0) + triu(w, 1)' + eye(N); % diagnoal equals 1
theta = rand(N,1);

% clamped statistics
s_c = sum(s)/ P;
ss_c = (s' * s)/P;

% random sequential dynamics
S = zeros(T,N);
dw = zeros(K,1);
un_p = zeros(T,1);
L = zeros(K,1);

for i = 1: K
    
    % random sequential dynamics
    m = 2 * randi([0,1],1,N) - 1;
    % m = s_c;
    
    for t = 1: T
        
%         temp = w' * repmat(double(m),10,1);
%         m = tanh(temp(1,:) + theta');
        % m = (int8(m>0.5)*2)-1;
        idx = randi(10);
        m(idx) = -1*m(idx);
        un_p(t) = un_boltzmann_dist(w,theta,m);
        S(t,:) = m;
    end
%     s_f = sum(S)/P;
%     ss_f = (S'*S)/P;
    un_p = un_p/(sum(un_p(:)));
    temp = repmat(un_p',N,1);
    s_f = S'*un_p;
    ss_f = S'*(S.*temp');
    
    theta_grad = s_c - s_f';
    w_grad = ss_c - ss_f;
    theta = theta - eta*theta_grad';
    w = w - eta*w_grad;
    
    dw(i) = (mean(w_grad(:)));
    p_sum = 0;
    for j = 1:P
        p_now = un_boltzmann_dist(w,theta,s(j,:));
        p_sum = p_sum+log(p_now);
    end
    L(i) = 1/P*p_sum;
    
end

figure(1);
% scatter(1:length(dw),dw)
dw = dw- min(dw);
xlim([1,length(dw)])
plot(dw)
title("change of the weights along the learning")

figure(2)
plot(L)
title("change of the log likelihood function along the learning")

        