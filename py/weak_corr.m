close all; clf; clear;
fileID = fopen('bint.txt','r');
formatSpec = '%d';
data = fscanf(fileID,formatSpec);
fclose(fileID);
n_neurons = 160;
n_repeats = 297;
n_timebins = 953;
T = n_timebins*20/1000; % s
len = n_repeats*n_timebins;
data = reshape(data,n_neurons,n_repeats*n_timebins);



%% Assignment
N = 10; % number of cells
new_data = data(1:10,:);
patterns = dec2bin(2^N-1:-1:0)-'0'; %1024*10
patterns_count = zeros(2^N,1);
% calculate rate of each pattern
for i = 1:len
    pattern = new_data(:,i);
    idx = find(ismember(patterns,pattern','row'));
    patterns_count(idx) = patterns_count(idx) + 1;
end
r_prob = patterns_count/(sum(patterns_count));
r_rate = patterns_count/T;

% calculate the predicted pattern rates by assuming independence
% prob of spike for each cell
p_spike = sum(new_data')/len; % 1*10
p_nspike = 1-p_spike;
p_patterns = repmat(p_spike,2^N,1).*patterns;
for i=1:2^N
    pp = p_patterns(i,:);
    for j=1:N
        if pp(j)==0
            pp(j)=p_nspike(j);
        end
    end
    p_patterns(i,:)=pp;
end
ind_prob = prod(p_patterns');





% maximum entropy model
% assuming h(N*1) and J(N*N) comes from a uniformly generated distrbution
mu_h = -1;
sigma_h = 0.5;
mu_J = 0;
sigma_J = 0.1;
h = normrnd(mu_h,sigma_h,[N,1]);
J = normrnd(mu_J, sigma_J,[N,N]);

a_patterns = patterns*2-1; % modify the pattern to -1/1
P2 = zeros(2^N,1);
for i = 1:2^N
    pattern = a_patterns(i,:);%1*N
    p(i) = exp(pattern*h+0.5* sum(sum(((J-diag(J)).*(pattern'*pattern)))));    
end

p = p/sum(p);


figure(1)

loglog(r_prob,ind_prob',"o")
hold on
loglog(r_prob,p,'*')
loglog(r_prob,r_prob,'-')

xlabel("observed pattern probability")
ylabel("approximated pattern probability")



